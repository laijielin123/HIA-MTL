##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path, precision, sensitivity, \
    specificity
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
import logging

# 元训练阶段
class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        # 设置了保存记录和检查点的文件夹路径
        log_base_dir = '/root/meta-transfer-learning/logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        # 在log_base_dir其中创建了一个名为 meta 的子文件夹
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
            '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        # 将参数 args 存储在类的实例变量 self.args 中，以便在类中共享使用
        self.args = args

        # Load meta-train set
        # 加载了用于元训练的数据集，并创建了一个数据加载器 train_loader
        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Load meta-val set
        # 加载了用于验证的元验证集，并创建了一个数据加载器 val_loader
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        
        # Build meta-transfer learning model
        # 构建元迁移学习模型
        self.model = MtlLearner(self.args)

        # Set optimizer 
        # 设置了优化器，使用的是 Adam 优化器。优化器对两个参数组进行优化
        # 模型的不同部分可以有不同的学习率 
        # 第一组：包含 encoder（编码器）部分的参数，只选择那些需要更新的参数（requires_grad 为 True）。这些参数使用基础学习率 self.args.meta_lr1
        # 第二组：包含 base_learner（基础学习器）部分的参数，指定它们的学习率为 self.args.meta_lr2
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler 
        # 设置了学习率调度器，使用的是 StepLR 方法。调度器将在每个 step_size 步骤后，将学习率乘以一个衰减因子gamma。
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        # load pretrained model without FC classifier 加载预训练模型（不包含全连接分类器）
        # 获取当前模型的状态字典（参数字典）
        self.model_dict = self.model.state_dict()
        # 如果 self.args.init_weights 不为 None，则加载指定路径的预训练模型权重文件，并提取其中的 params（参数）
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        # 如果 self.args.init_weights 为 None，则构建默认路径 pre_save_path，加载默认路径下的预训练模型权重文件，并提取其中的 params（参数）
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
                str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        # 为预训练模型的参数名称添加 encoder. 前缀，并过滤掉不在当前模型状态字典中的参数
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        # 打印加载的预训练参数名称，用于调试
        print(pretrained_dict.keys())
        # 使用预训练参数更新当前模型的状态字典，并加载更新后的状态字典
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)    

        # Set model to GPU 将模型设置为使用GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    # 保存模型函数
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        # 设置元训练日志
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['train_pre'] = []
        trlog['val_pre'] = []
        trlog['train_sen'] = []
        trlog['val_sen'] = []
        trlog['train_spec'] = []
        trlog['val_spec'] = []
        
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        trlog['max_pre'] = 0.0
        trlog['max_sen'] = 0.0
        trlog['max_spec'] = 0.0
        trlog['max_pre_epoch'] = 0
        trlog['max_sen_epoch'] = 0
        trlog['max_spec_epoch'] = 0

        # 设置计时器和全局计数
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        # 用于将训练和验证过程中的数据记录到 Tensorboard，以便可视化
        writer = SummaryWriter(comment=self.args.save_path)

        # Generate the labels for train set of the episodes
        # 生成了训练集的标签 label_shot。这些标签用于元训练中的支持集（shot set），并将其转换为适合GPU计算的 LongTensor 类型
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
        
        # 初始化连续准确率不增长的计数器
        no_improvement_count = 0
        best_accuracy = 0  # 用于记录最佳准确率
        
        # Start meta-train
        # 开始元训练循环，遍历从第1到最大epoch
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            # 在每个epoch开始时，更新学习率调度器
            self.lr_scheduler.step()
            # Set the model to train mode
            # 将模型设置为训练模式
            self.model.train()
            # Set averager classes to record training losses and accuracies
            # 初始化两个 Averager 类，用于记录训练损失和准确率的平均值
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_pre_averager = Averager()
            train_sen_averager = Averager()
            train_spec_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates
            # 生成测试集的标签 label，用于元训练中的查询集（query set），并将其转换为适合GPU计算的 LongTensor 类型
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Using tqdm to read samples from train loader
            # 使用 tqdm 读取训练数据，显示进度条，并更新全局计数
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                # 将数据和标签移至GPU（如果可用）。然后，根据 shot 和 way 参数，将数据分为支持集（shot set）和查询集（query set）
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                # Output logits for model
                # 使用模型对支持集和查询集进行预测
                logits = self.model((data_shot, label_shot, data_query))
                # Calculate meta-train loss
                # 计算元训练交叉熵损失
                loss = F.cross_entropy(logits, label)
                # Calculate meta-train accuracy
                # 计算元训练准确率
                acc = count_acc(logits, label)
                pre = precision(label, logits)
                sen = sensitivity(label, logits)
                spec = specificity(label, logits)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                writer.add_scalar('data/pre', float(pre), global_count)
                writer.add_scalar('data/sen', float(sen), global_count)
                writer.add_scalar('data/spec', float(spec), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description(
                    'Epoch {}, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format(epoch, loss.item(), acc,
                                                                                                pre, sen, spec))

                # Add loss and accuracy for the averagers
                # 将损失和准确率添加到 Averager 中
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                train_pre_averager.add(pre)
                train_sen_averager.add(sen)
                train_spec_averager.add(spec)

                # Loss backwards and optimizer updates
                # 并执行反向传播和优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            # 将 Averager 的值转换为数值
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            train_pre_averager = train_pre_averager.item()
            train_sen_averager = train_sen_averager.item()
            train_spec_averager = train_spec_averager.item()

            # Start validation for this epoch, set model to eval mode
            # 开始验证，将模型设置为评估模式
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            # 初始化验证损失和准确率的 Averager
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_pre_averager = Averager()
            val_sen_averager = Averager()
            val_spec_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch
            # 生成验证集的标签，并将其转换为适合GPU计算的 LongTensor 类型
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
                
            # Print previous information
            # 每10个epoch打印一次当前最高的验证准确率和对应的epoch
            if epoch % 10 == 0:
                print('Best Val acc={:.4f} Best Val pre={:.4f} Best Val sen={:.4f} Best Val spec={:.4f} '
                      .format(trlog['max_acc'], trlog['max_pre'], trlog['max_sen'], trlog['max_spec']))
            
            # val_acc_list = []
            
            # Run meta-validation
            # 进行元验证，计算损失和准确率，并更新验证损失和准确率的 Averager。
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                pre = precision(label, logits)
                sen = sensitivity(label, logits)
                spec = specificity(label, logits)

                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)
                val_pre_averager.add(pre)
                val_sen_averager.add(sen)
                val_spec_averager.add(spec)
                # val_acc_list.append(acc)

            # Update validation averagers
            # 验证损失和准确率更新为数值
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_pre_averager = val_pre_averager.item()
            val_sen_averager = val_sen_averager.item()
            val_spec_averager = val_spec_averager.item()
            # Write the tensorboardX records 并记录到 Tensorboard
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            writer.add_scalar('data/val_pre', float(val_pre_averager), epoch)
            writer.add_scalar('data/val_sen', float(val_sen_averager), epoch)
            writer.add_scalar('data/val_spec', float(val_spec_averager), epoch)
            # Print loss and accuracy for this epoch  打印当前epoch的验证损失和准确率
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format
                  (epoch, val_loss_averager, val_acc_averager, val_pre_averager, val_sen_averager, val_spec_averager))

            if val_acc_averager > best_accuracy:
                best_accuracy = val_acc_averager
                no_improvement_count = 0  # 重置计数器
            else:
                no_improvement_count += 1

                # 如果连续20个 epoch 准确率没有提升，则终止训练
            if no_improvement_count >= 50:
                print("Early stopping: Accuracy hasn't improved for 20 epochs.")
                break
            # Update best saved model
            # 如果当前epoch的验证准确率超过之前的最高值，则更新最高准确率，并保存当前模型。同时，每10个epoch保存一次模型
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            if val_pre_averager > trlog['max_pre']:
                trlog['max_pre'] = val_pre_averager
                trlog['max_pre_epoch'] = epoch
                self.save_model('max_pre')

            if val_sen_averager > trlog['max_sen']:
                trlog['max_sen'] = val_sen_averager
                trlog['max_sen_epoch'] = epoch
                self.save_model('max_sen')

            if val_spec_averager > trlog['max_spec']:
                trlog['max_spec'] = val_spec_averager
                trlog['max_spec_epoch'] = epoch
                self.save_model('max_spec')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            # 更新日志并保存
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['train_pre'].append(train_pre_averager)
            trlog['train_sen'].append(train_sen_averager)
            trlog['train_spec'].append(train_spec_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            trlog['val_pre'].append(val_pre_averager)
            trlog['val_sen'].append(val_sen_averager)
            trlog['val_spec'].append(val_spec_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
            
#             # Hard Task Meta-Batch 策略
#             hard_task_indices = np.argsort(val_acc_list)[:3]  # 选择准确度最低的 10 个任务
#             self.model.train()  # 确保模型处于训练模式

#             # Adjust learning rate for hard task training
#             original_lr = self.optimizer.param_groups[0]['lr']
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] *= 0.5  # Decrease learning rate by a factor of 10 (adjust as needed)

#             for idx in hard_task_indices:
#                 hard_task_data, hard_task_label = self.val_sampler.sample_task_by_index(idx, self.valset)
#                 if torch.cuda.is_available():
#                     hard_task_data = hard_task_data.cuda()
#                     hard_task_label = hard_task_label.cuda()

#                 # print("hard_task_data shape:", hard_task_data.shape)
#                 # print("hard_task_label shape:", hard_task_label.shape)

#                 p = self.args.shot * self.args.way
#                 data_shot, data_query = hard_task_data[:p], hard_task_data[p:]

#                 # print("data_shot shape:", data_shot.shape)
#                 # print("data_query shape:", data_query.shape)

#                 logits = self.model((data_shot, hard_task_label[:p], data_query))
#                 loss = F.cross_entropy(logits, hard_task_label[p:])
#                 acc = count_acc(logits, hard_task_label[p:])
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 # print(f"Task {idx}: loss = {loss.item()}, acc = {acc}")
#             # Restore original learning rate after hard task training
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = original_lr

        writer.close()

    # 用于（meta）元测试和元评估阶段
    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        # 加载日志
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        # Load meta-test set
        # 加载元测试集
        test_set = Dataset('test', self.args)
        # 使用 CategoriesSampler 进行采样，每次采样600个任务，每个任务包含way种类，每种类shot + val_query张图片。
        sampler = CategoriesSampler(test_set.label, 200, self.args.way, self.args.shot + self.args.val_query)
        # # 将测试集加载到 DataLoader 中，并使用多个工作线程以加快数据加载
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        # 初始化一个数组 test_acc_record 来记录每个任务的测试准确率，共600个任务
        test_acc_record = np.zeros((200,))
        test_pre_record = np.zeros((200,))
        test_sen_record = np.zeros((200,))
        test_spec_record = np.zeros((200,))

        # Load model for meta-test phase
        # 加载用于元测试阶段的模型
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        # 将模型设置为评估模式，以确保在推理时不进行参数更新和丢弃层（dropout）
        self.model.eval()

        # Set accuracy averager
        # 初始化一个 Averager 对象 ave_acc，用于记录和计算测试过程中准确率的平均值
        ave_acc = Averager()
        max_acc = Averager()
        ave_pre = Averager()
        ave_sen = Averager()
        ave_spec = Averager()

        # Generate labels
        # 生成测试样本[查询集（val_query）]和训练样本[支持集（shot）]的标签，并将其转换为适合GPU计算的 LongTensor 类型
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
            
        # 设置日志文件的路径
        logfile = '/root/meta-transfer-learning/logs/all_logs/filetext.log'

        # 配置日志记录
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        logging.info(f'test\n')
        max = 0
        # Start meta-test
        # 开始元测试
        # 对每个批次的任务进行评估
        for i, batch in enumerate(loader, 1):
            # 移至GPU（如果可用），并将数据分为训练样本（支持集）和测试样本（查询集）
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            # 将数据使用模型对数据进行预测，计算准确率，并记录在 ave_acc 和 test_acc_record 中
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
            pre = precision(label, logits)
            sen = sensitivity(label, logits)
            spec = specificity(label, logits)
            ave_acc.add(acc)
            max_acc.add(acc)
            ave_pre.add(pre)
            ave_sen.add(sen)
            ave_spec.add(spec)
            test_acc_record[i-1] = acc
            test_pre_record[i - 1] = pre
            test_sen_record[i - 1] = sen
            test_spec_record[i - 1] = spec
            # 每处理100个任务，打印当前的平均准确率和当前任务的准确率。
            if i % 40 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
                if max_acc.item() > max:
                    max = max_acc.item()
                max_acc.reset()
            
            # Evaluate on hard tasks
        # val_acc_list = []
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = self.args.shot * self.args.way
            data_shot, data_query = data[:p], data[p:]
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
#             val_acc_list.append(acc)

#         hard_task_indices = np.argsort(val_acc_list)[:3]  # Select 10 tasks with lowest accuracy
#         hard_task_accs = []
#         for idx in hard_task_indices:
#             hard_task_data, hard_task_label = self.val_sampler.sample_task_by_index(idx, test_set)
#             if torch.cuda.is_available():
#                 hard_task_data = hard_task_data.cuda()
#                 hard_task_label = hard_task_label.cuda()
#             p = self.args.shot * self.args.way
#             data_shot, data_query = hard_task_data[:p], hard_task_data[p:]
#             logits = self.model((data_shot, label_shot, data_query))
#             acc = count_acc(logits, label)
#             hard_task_accs.append(acc)
        
#         hard_task_avg_acc = np.mean(hard_task_accs)
#         print('Hard Task Average Accuracy: {:.2f}'.format(hard_task_avg_acc * 100))
            
        # Calculate the confidence interval, update the logs
        # 计算置信区间并更新日志
        # 计算测试准确率的置信区间（compute_confidence_interval），并打印验证过程中最佳的epoch及其对应的准确率、当前测试的平均准确率及其置信区间。
        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test Acc {:.4f}'.format(ave_acc.item()))
        logging.info('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        # print('Hard Task Acc {:.4f}'.format(hard_task_avg_acc))

        m, pm = compute_confidence_interval(test_pre_record)
        print('Val Best Epoch {}, pre {:.4f}, Test pre {:.4f}'.format(trlog['max_pre_epoch'], trlog['max_pre'],
                                                                      ave_pre.item()))
        print('Test pre {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test pre {:.4f}'.format(ave_pre.item()))
        logging.info('Test pre {:.4f} + {:.4f}'.format(m, pm))

        m, pm = compute_confidence_interval(test_sen_record)
        print('Val Best Epoch {}, sen {:.4f}, Test sen {:.4f}'.format(trlog['max_sen_epoch'], trlog['max_sen'],
                                                                      ave_sen.item()))
        print('Test sen {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test sen {:.4f}'.format(ave_sen.item()))
        logging.info('Test sen {:.4f} + {:.4f}'.format(m, pm))

        m, pm = compute_confidence_interval(test_spec_record)
        print('Val Best Epoch {}, spec {:.4f}, Test spec {:.4f}'.format(trlog['max_spec_epoch'], trlog['max_spec'],
                                                                        ave_spec.item()))
        print('Test spec {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test spec {:.4f}'.format(ave_spec.item()))
        logging.info('Test spec {:.4f} + {:.4f}'.format(m, pm))
        