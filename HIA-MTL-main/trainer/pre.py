##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """  # 预阶段的培训师。
import logging
import os.path as osp
import numpy as np
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path, precision, sensitivity, specificity
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""  # 包含预训练阶段代码的类
    def __init__(self, args):
        # Set the folder to save the records and checkpoints 设置文件夹以保存记录和检查点

        log_base_dir = '/root/meta-transfer-learning/logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class 将参数设置为可在类中共享
        self.args = args

        # Load pretrain set 加载预训练集
        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Load meta-val set 加载元数据集
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Set pretrain class number  设置预训练课程编号
        num_class_pretrain = self.trainset.num_class  # 两类

        # Build pretrain model 建立预训练模型
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)

        # Set optimizer 设置优化器,执行梯度下降优化算法
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
            {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)
        # Set learning rate scheduler 设置学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)
        
        # Set model to GPU 将模型设置为GPU

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            print("用的GPU")
        else:
            print("GPU占用高，用的CPU")

        
    def save_model(self, name):
        """The function to save checkpoints. 保存检查点的函数。
        Args:
          name: the name for saved checkpoint已保存检查点的名称
        """  
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))
        
    def train(self):
        """The function for the pre-train phase."""  # 训练运行前阶段的功能

        # Set the pretrain log 设置预训练日志
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
        trlog['train_AUC'] = []
        trlog['val_AUC'] = []
        # trlog['train_F1'] = []
        # trlog['val_F1'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        trlog['max_pre'] = 0.0
        trlog['max_sen'] = 0.0
        trlog['max_spec'] = 0.0
        trlog['max_AUC'] = 0.0
        # trlog['max_F1'] = 0.0
        trlog['max_pre_epoch'] = 0
        trlog['max_sen_epoch'] = 0
        trlog['max_spec_epoch'] = 0
        trlog['max_AUC_epoch'] = 0
        # trlog['max_F1_epoch'] = 0

        logfile = f'/root/meta-transfer-learning/logs/all_logs/{self.args.dataset}_pre_{self.args.shot}shot.log'
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')

        # Set the timer 设置计时器
        timer = Timer()
        # Set global count to zero 将全局计数设置为零
        global_count = 0
        # Set tensorboardX 设置tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain 开始预训练
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate 更新学习率
            self.lr_scheduler.step()
            # Set the model to train mode 将模型设置为训练模型
            self.model.train()
            self.model.mode = 'pre'
            # print("train")
            # print(self.model)
            # Set averager classes to record training losses and accuracies 设置平均类以记录训练损失和准确性
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_pre_averager = Averager()
            train_sen_averager = Averager()
            train_spec_averager = Averager()
            train_AUC_averager = Averager()
            # train_F1_averager = Averager()

            # start_memory = torch.cuda.memory_allocated()  # 记录开始时内存使用

            # Using tqdm to read samples from train loader 用tqdm读取装载机样本
            tqdm_gen = tqdm.tqdm(self.train_loader)

            for i, batch in enumerate(tqdm_gen, 1):  # i代表当前批次的索引，从1开始，batch当前批次的数据【0】为data【1】为label
                # Update global count number 更新全局计数
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                    # print("在GPU")
                else:
                    # print("zaiCPU")
                    data = batch[0]
                label = batch[1]
                # batchsize为256其为256为128其为128 print(f"data={len(data)}")
                # 同上 print(f"label={len(label)}")
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                # Output logits for model 输出模型的logits

                logits = self.model(data)  # logits和label都是128
                # 同上print(f"logits={len(logits)}")
                # Calculate train loss计算训练损失
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy 计算训练精度
                acc = count_acc(logits, label)
                pre = precision(label, logits)
                sen = sensitivity(label, logits)
                spec = specificity(label, logits)
                # label1 = label.to('cpu').numpy()  # zhuan numpy shuzu
                # logits1 = logits.detach().to('cpu').numpy()
                # predicted_labels = np.argmax(logits1, axis=1)
                # try:
                #
                #     AUC = roc_auc_score(label1, logits1[:, 1])
                # except ValueError as e:
                #     print(f"error:{e}")
                #     pass
                # F1 = f1_score(label1, np.argmax(logits1, axis=1))

                # Write the tensorboardX records 编写tensorboardX记录
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                writer.add_scalar('data/pre', float(pre), global_count)
                writer.add_scalar('data/sen', float(sen), global_count)
                writer.add_scalar('data/spec', float(spec), global_count)
                # writer.add_scalar('data/AUC', float(AUC), global_count)
                # writer.add_scalar('data/F1', float(F1), global_count)
                # Print loss and accuracy for this step 此步骤的打印损失和准确性

                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format(epoch, loss.item(), acc, pre, sen, spec))

                # Add loss and accuracy for the averagers 增加平均值的损失和准确性
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                train_pre_averager.add(pre)
                train_sen_averager.add(sen)
                train_spec_averager.add(spec)
                # train_AUC_averager.add(AUC)
                # train_F1_averager.add(F1)


                # Loss backwards and optimizer updates 损失向后和优化器更新  梯度下降算法进行优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers 更新平均值
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            train_pre_averager = train_pre_averager.item()
            train_sen_averager = train_sen_averager.item()
            train_spec_averager = train_spec_averager.item()
            # train_AUC_averager = train_AUC_averager.item()
            # train_F1_averager = train_F1_averager.item()

            # Start validation for this epoch, set model to eval mode 开始验证此epoch，将模型设置为eval模式
            self.model.eval()
            self.model.mode = 'preval'
            # print("train_val")
            # print(self.model)
            # 可以到这边print("成功了")
            # Set averager classes to record validation losses and accuracies 设置平均值类别以记录验证损失和准确性
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_pre_averager = Averager()
            val_sen_averager = Averager()
            val_spec_averager = Averager()
            # val_AUC_averager = Averager()
            # val_F1_averager = Averager()

            # Generate the labels for test 生成测试标签
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            # print(f"label={label}")
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)  # self.args.way =5,表示一个任务（task）中的类别数。
            # 这部分代码生成了一个从 0 到 self.args.way-1 的整数序列.将生成的整数序列重复 self.args.shot 次。self.args.shot
            # 表示在每个任务的支持集中包含每个类别的样本数量。您可以为每个类别生成 self.args.shot=1 个相同的标签，以构建支持集中的标签。
            # 当way为2的时候label_shot为30 print(f"label_shot为={label_shot}")
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)

            # Print previous information  打印以前的信息
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f} Best Val pre={:.4f} Best Val sen={:.4f} Best Val spec={:.4f} '
                      .format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['max_pre'],
                                                                      trlog['max_sen'], trlog['max_spec']))
                logging.info('Best Epoch {}, Best Val acc={:.4f} Best Val pre={:.4f} Best Val sen={:.4f} Best Val spec={:.4f} '
                      .format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['max_pre'],
                                                                      trlog['max_sen'], trlog['max_spec']))
            # Run meta-validation 运行元验证
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                # len(data)为32 print(f"data的len为{len(data)}")
                data_shot, data_query = data[:p], data[p:]
                # 此时data_shot为2 print(f"data_shot={len(data_shot)}")
                # 此时data_query为30 print(f"data_query={len(data_query)}")
                logits = self.model((data_shot, label_shot, data_query))  # 进入self.model.preval_forward
                # print(f"logits{len(logits)}")
                # 此时logits与label都为30 print(f"label{len(label)}")
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                pre = precision(label, logits)
                sen = sensitivity(label, logits)
                spec = specificity(label, logits)
                # label1 = label.to('cpu').numpy()  # zhuan numpy shuzu
                # logits1 = logits.detach().to('cpu').numpy()
                # # predicted_labels = np.argmax(logits1, axis=1)
                # try:
                #
                #     AUC = roc_auc_score(label1, logits1[:, 1])
                # except ValueError as e:
                #     print(f"error:{e}")
                #     pass
                # F1 = f1_score(label1, np.argmax(logits1, axis=1))

                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)
                val_pre_averager.add(pre)
                val_sen_averager.add(sen)
                val_spec_averager.add(spec)
                # val_AUC_averager.add(AUC)
                # val_F1_averager.add(F1)

            # Update validation averagers 更新验证平均值
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_pre_averager = val_pre_averager.item()
            val_sen_averager = val_sen_averager.item()
            val_spec_averager = val_spec_averager.item()
            # val_AUC_averager = val_AUC_averager.item()
            # val_F1_averager = val_F1_averager.item()

            # Write the tensorboardX records 编写tensorboardX记录
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            writer.add_scalar('data/val_pre', float(val_pre_averager), epoch)
            writer.add_scalar('data/val_sen', float(val_sen_averager), epoch)
            writer.add_scalar('data/val_spec', float(val_spec_averager), epoch)
            # writer.add_scalar('data/val_AUC', float(val_AUC_averager), epoch)
            # writer.add_scalar('data/val_F1', float(val_F1_averager), epoch)
            # Print loss and accuracy for this epoch
            # end_memory = torch.cuda.memory_allocated()  # 循环结束时内存占用
            # max_memory = torch.cuda.max_memory_allocated()  # 整个epoch最大内存分配
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format
                  (epoch, val_loss_averager, val_acc_averager, val_pre_averager, val_sen_averager,val_spec_averager))
            logging.info('Epoch {}, Val, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format
                  (epoch, val_loss_averager, val_acc_averager, val_pre_averager, val_sen_averager,val_spec_averager))
            # logging.info(
            #     f'Epoch {epoch}: Start memory: {start_memory}, End memory: {end_memory}, Max memory: {max_memory}')
            # Update best saved model 更新最佳保存模型
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')

            # if val_pre_averager > trlog['max_pre']:
            #     trlog['max_pre'] = val_pre_averager
            #     trlog['max_pre_epoch'] = epoch
            #     self.save_model('max_pre')
            #
            # if val_sen_averager > trlog['max_sen']:
            #     trlog['max_sen'] = val_sen_averager
            #     trlog['max_sen_epoch'] = epoch
            #     self.save_model('max_sen')
            #
            # if val_spec_averager > trlog['max_spec']:
            #     trlog['max_spec'] = val_spec_averager
            #     trlog['max_spec_epoch'] = epoch
            #     self.save_model('max_spec')
            #
            # if val_AUC_averager > trlog['max_AUC']:
            #     trlog['max_AUC'] = val_AUC_averager
            #     trlog['max_AUC_epoch'] = epoch
            #     self.save_model('max_AUC')
            #
            # if val_F1_averager > trlog['max_F1']:
            #     trlog['max_F1'] = val_F1_averager
            #     trlog['max_F1_epoch'] = epoch
            #     self.save_model('max_F1')
            # Save model every 10 epochs 每10个时期保存一次模型
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs 更新日志
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['train_pre'].append(train_pre_averager)
            trlog['train_sen'].append(train_sen_averager)
            trlog['train_spec'].append(train_spec_averager)
            # trlog['train_AUC'].append(train_AUC_averager)
            # trlog['train_F1'].append(train_F1_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            trlog['val_pre'].append(val_pre_averager)
            trlog['val_sen'].append(val_sen_averager)
            trlog['val_spec'].append(val_spec_averager)
            # trlog['val_AUC'].append(val_AUC_averager)
            # trlog['val_F1'].append(val_F1_averager)

            # Save log 保存日志
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
                # logging.info('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        writer.close()

