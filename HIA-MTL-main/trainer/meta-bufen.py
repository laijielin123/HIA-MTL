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
import logging
import shutil
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from dataloader.samplers import FailedCategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
from dataloader.dataset_loader import FailedDatasetLoader

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

        # 创建一个用于存储失败样本的目标文件夹
        target_folder = '/root/dataset/BreakHis/5fen/40X5-224/HT_val'
        os.makedirs(target_folder, exist_ok=True)
        # Set the meta-train log
        # 设置元训练日志
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['val_HT_loss'] = []
        trlog['val_HT_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # 设置计时器和全局计数
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        # 用于将训练和验证过程中的数据记录到 Tensorboard，以便可视化
        writer = SummaryWriter(comment=self.args.save_path)

        # 初始化连续准确率不增长的计数器
        no_improvement_count = 0
        best_accuracy = 0  # 用于记录最佳准确率

        # Start meta-train
        # 开始元训练循环，遍历从第1到最大epoch
        for epoch in range(1, self.args.max_epoch + 1):
            
            failed_samples_path = []
            failed_samples_count = {}

            failed_samples_paths = []
            # Generate the labels for train set of the episodes
            # 生成了训练集的标签 label_shot。这些标签用于元训练中的支持集（shot set），并将其转换为适合GPU计算的 LongTensor 类型
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)

            # Update learning rate
            # 在每个epoch开始时，更新学习率调度器
            self.lr_scheduler.step()
            # Set the model to train mode
            # 将模型设置为训练模式
            self.model.train()
            self.model.mode = "preval"
            # Set averager classes to record training losses and accuracies
            # 初始化两个 Averager 类，用于记录训练损失和准确率的平均值
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            train_HT_loss_averager = Averager()
            train_HT_acc_averager = Averager()

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

            if epoch > 1:
                # 每10个epoch，清空HT_val文件夹中的所有文件
                for filename in os.listdir(target_folder):
                    file_path = os.path.join(target_folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                # # 将文件路径添加到 failed_samples_paths 列表中,HT硬任务开始
                for path, _ in filtered_failed_samples:
                    failed_samples_paths.append(path)

                for file_path in failed_samples_paths:

                    # 提取文件路径中的倒数第二个文件夹的名称作为新文件夹的名称
                    folder_name = os.path.basename(os.path.dirname(file_path))

                    # 创建真实标签对应的文件夹路径
                    true_label_folder = os.path.join(target_folder, folder_name)
                    os.makedirs(true_label_folder, exist_ok=True)

                    # 构建新的文件名，以索引号命名
                    new_file_name = os.path.basename(file_path)

                    # 目标路径为真实标签文件夹路径下的新文件名
                    target_path = os.path.join(true_label_folder, new_file_name)

                    # 如果目标路径不存在该文件，则复制文件到目标路径
                    if not os.path.exists(target_path):
                        shutil.copy(file_path, target_path)

                # 检查每个子文件夹中的文件数量
                subfolders = os.listdir(target_folder)
                # 初始化最小文件数量为一个足够大的数
                min_file_count = 1000
                filtered_failed_samples.clear()
                # 检查每个子文件夹中的文件数量
                for subfolder in subfolders:
                    subfolder_path = os.path.join(target_folder, subfolder)
                    if os.path.isdir(subfolder_path):  # 确保是文件夹而不是文件
                        file_count = len(os.listdir(subfolder_path))
                        print(f'epoch+{file_count}')
                        if file_count < 6:
                            shutil.rmtree(subfolder_path)
                            print('shanle')
                        elif file_count < min_file_count:
                            min_file_count = file_count
                if min_file_count == 1000:
                    min_file_count = 1
                subfolders = os.listdir(target_folder)
                print(len(subfolders))
            # 从这边运行HT
            if epoch > 120:
                if len(subfolders) >= 3:
                    label_shot2 = torch.arange(self.args.way).repeat(3)
                    if torch.cuda.is_available():
                        label_shot2 = label_shot2.type(torch.cuda.LongTensor)
                    else:
                        label_shot2 = label_shot2.type(torch.LongTensor)

                    # Load meta-val set 加载硬任务验证集
                    # print('Jin HT')

                    self.HT_valset = Dataset('HT_val', self.args)
                    # print(f'label={self.HT_valset.label}')
                    self.HT_val_sampler = CategoriesSampler(self.HT_valset.label, 200, self.args.way,
                                                            6)
                    self.HT_val_loader = DataLoader(dataset=self.HT_valset, batch_sampler=self.HT_val_sampler,
                                                    num_workers=8,
                                                    pin_memory=True)

                    label2 = torch.arange(self.args.way).repeat(3)
                    if torch.cuda.is_available():
                        label2 = label2.type(torch.cuda.LongTensor)
                    else:
                        label2 = label2.type(torch.LongTensor)

                    for i, batch in enumerate(self.HT_val_loader, 1):  # 验证数据加载器中的批次数据
                        # print(f'batch:{batch}')  # 将数据移动到 GPU
                        if torch.cuda.is_available():
                            data, _ = [_.cuda() for _ in batch]
                        else:
                            data = batch[0]
                        p2 = self.args.way * 3
                        data_shot, data_query = data[:p2], data[p2:]
                        HT_logits = self.model((data_shot, label_shot2, data_query))
                        HT_loss = F.cross_entropy(HT_logits, label2) # 使用交叉熵损失计算预测困难任务损失
                        HT_acc = count_acc(HT_logits, label2)

                        # 记录预测错误的样本
                        predictions = HT_logits.argmax(dim=1)  # 获取预测值
                        incorrect_indices = (predictions != label2).nonzero()[:, 0]  # 找到预测错误的样本索引
                        # 遍历错误的样本索引，获取对应的真实标签和文件路径，并记录这些信息。如果某个文件路径已经记录过，则增加其计数，否则初始化为 1
                        for index in incorrect_indices:
                            index = int(index)
                            true_label = self.train_sampler.H_batch[index]
                            true_label = int(true_label)
                            file_path = self.trainset.label_to_data.get(true_label)

                            if file_path is not None:
                                failed_samples_path.append(file_path)
                                if file_path in failed_samples_count:
                                    failed_samples_count[file_path] += 1
                                else:
                                    failed_samples_count[file_path] = 1
                            else:
                                print(f"Failed to find file path for true label: {true_label}")

                        # Print loss and accuracy for this step 此步骤的打印损失和准确性
                        tqdm_gen.set_description(
                            'Epoch {}, HT_Loss={:.4f} HT_Acc={:.4f}'.format(epoch, HT_loss.item(), HT_acc))

                        # Add loss and accuracy for the averagers 增加困难任务平均值的损失和准确性
                        train_HT_loss_averager.add(HT_loss.item())
                        train_HT_acc_averager.add(HT_acc)

                        # Loss backwards and optimizer updates 损失向后和优化器更新
                        self.optimizer.zero_grad()
                        HT_loss.backward()
                        self.optimizer.step()
                    print("HT过了")
                    # # 筛选出现次数大于等于八的文件
                    #
                    # filtered_failed_samples = [(file_path, count) for file_path, count in
                    #                            failed_samples_count.items() if count >= 4]

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

                predictions = logits.argmax(dim=1)  # 获取预测值
                incorrect_indices = (predictions != label).nonzero()[:, 0]  # 找到预测错误的样本索引

                for index in incorrect_indices:
                    index = int(index)
                    true_label = self.train_sampler.batch[index]
                    true_label = int(true_label)
                    file_path = self.trainset.label_to_data.get(true_label)

                    if file_path is not None:
                        failed_samples_path.append(file_path)
                        if file_path in failed_samples_count:
                            failed_samples_count[file_path] += 1
                        else:
                            failed_samples_count[file_path] = 1
                    else:
                        print(f"Failed to find file path for true label: {true_label}")

                # 按照失败文件出现的次数进行排序

                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                # 将损失和准确率添加到 Averager 中
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                # 并执行反向传播和优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 筛选出现次数大于等于八的文件
            # 实现一个动态的困难样本筛选机制，随着训练的进行，逐渐放宽筛选标准，以便模型能够更多地关注那些难以正确预测的样本
            # failed_samples_count字典存储了每个错误样本（file_path）及其出现次数（count）。sorted 函数通过一个 lambda 函数指定按照出现次数（x[1]）降序排序

            sorted_failed_samples = sorted(failed_samples_count.items(), key=lambda x: x[1], reverse=True)
            # filtered_failed_samples = [(file_path, count) for file_path, count in failed_samples_count.items() if
            #                            count >= 4]
            if epoch < 5:  # 有无动态HT
                filtered_failed_samples = [(file_path, count) for file_path, count in failed_samples_count.items() if
                                           count >= 4]
                print("4dang")

            if epoch >= 5 and epoch < 10:
                filtered_failed_samples = [(file_path, count) for file_path, count in failed_samples_count.items() if
                                           count >= 3]
                print("3dang")
            if epoch >= 10 and epoch < 15:
                filtered_failed_samples = [(file_path, count) for file_path, count in failed_samples_count.items() if
                                           count >= 2]
                print("2dang")
            if epoch >= 15:
                filtered_failed_samples = [(file_path, count) for file_path, count in failed_samples_count.items() if
                                           count >= 1]
                print("1dang")
            
            # Update the averagers
            # 将 Averager 的值转换为数值
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            # 开始验证，将模型设置为评估模式
            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            # 初始化验证损失和准确率的 Averager
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch
            # 生成验证集的标签，并将其转换为适合GPU计算的 LongTensor 类型
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
                
            # 设置日志文件的路径
            logfile = '/root/meta-transfer-learning/logs/all_logs/BreakHis5fen.log'

            # 配置日志记录
            logging.basicConfig(filename=logfile, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s: %(message)s')

            # Start meta-test 启动元测试
            if epoch > 108:
                logging.info(f'trainval: epoch{epoch}\n')

            # Print previous information
            # 每10个epoch打印一次当前最高的验证准确率和对应的epoch
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
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

                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            # 验证损失和准确率更新为数值
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records 并记录到 Tensorboard
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)       
            # Print loss and accuracy for this epoch  打印当前epoch的验证损失和准确率
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

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
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            # 更新日志并保存
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))

        writer.close()

    # 用于（meta）元测试和元评估阶段
    def eval(self):
        failed_samples_path = []
        failed_samples_count = {}
        failed_samples_paths = []

        """The function for the meta-eval phase."""
        # Load the logs
        # 加载日志
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        # Load meta-test set
        # 加载元测试集
        test_set = Dataset('test', self.args)
        # 使用 CategoriesSampler 进行采样，每次采样600个任务，每个任务包含way种类，每种类shot + val_query张图片。
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        # # 将测试集加载到 DataLoader 中，并使用多个工作线程以加快数据加载
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        # 初始化一个数组 test_acc_record 来记录每个任务的测试准确率，共600个任务
        test_acc_record = np.zeros((600,))

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
        logfile = '/root/meta-transfer-learning/logs/all_logs/BreakHis5fen.log'

        # 配置日志记录
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')

        # failed_samples = []  # 存储测试失败的样本
        # Start meta-test 启动元测试
        logging.info(f'test\n')
        
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

            predictions = logits.argmax(dim=1)  # 获取预测值
            incorrect_indices = (predictions != label).nonzero()[:, 0]  # 找到预测错误的样本索引
            # print(f'inc/r={incorrect_indices}')
            for index in incorrect_indices:
                index = int(index)
                # print(f'index/r={index}')
                true_label = sampler.batch[index]
                true_label = int(true_label)
                file_path = test_set.label_to_data.get(true_label)

                if file_path is not None:
                    failed_samples_path.append(file_path)
                    if file_path in failed_samples_count:
                        failed_samples_count[file_path] += 1
                    else:
                        failed_samples_count[file_path] = 1
                else:
                    print(f"Failed to find file path for true label: {true_label}")

            # 按照失败文件出现的次数进行排序

            sorted_failed_samples = sorted(failed_samples_count.items(), key=lambda x: x[1], reverse=True)

            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            # 每处理100个任务，打印当前的平均准确率和当前任务的准确率。
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
            
        # Calculate the confidence interval, update the logs
        # 计算置信区间并更新日志
        # 计算测试准确率的置信区间（compute_confidence_interval），并打印验证过程中最佳的epoch及其对应的准确率、当前测试的平均准确率及其置信区间。
        
        # print(sorted_failed_samples)
        logging.info('失败者:{}'.format(sorted_failed_samples))


        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test Acc {:.4f}'.format(ave_acc.item()))
        logging.info('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        