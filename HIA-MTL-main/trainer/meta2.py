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
""" Trainer for meta-train phase. """  # 元训练阶段的培训师
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader  # dataloader是PyTorch中用于加载数据的重要工具，通常用于数据批量加载和并行数据加载
from dataloader.samplers import CategoriesSampler
from dataloader.samplers import FailedCategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path, precision, sensitivity, \
    specificity
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
from dataloader.dataset_loader import FailedDatasetLoader
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import logging
import shutil
# 下边可视化
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""  # 包含元训练阶段和元评估阶段代码的类

    def __init__(self, args):
        # Set the folder to save the records and checkpoints设置文件夹以保存记录和检查点
        log_base_dir = '/root/meta-transfer-learning/logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
                     '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(
            args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
                     '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
                     '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
                     '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class 将参数设置为可在类中共享
        self.args = args

        # Load meta-test set 加载元训练组
        self.trainset = Dataset('train', self.args)  # 先传给dataset——loader.py预处理获取到两个返回值，一个预处理后的图像数据一个标签
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way,
                                               self.args.shot + self.args.train_query)
        # 返回一个迭代器，通过迭代器可以获得一系列 episodic 数据集的任务样本索引。episodic指的是例如，一个 5-way 1-shot 分类任务意味着每个 episodic 任务包含 5 个类别，
        # 每个类别只包含 1 个样本。这个任务由支持集和查询集组成，其中支持集包含query个样本，每个类别一个，查询集包含更多的样本，通常也包括 5 个类别。
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8,
                                       pin_memory=True)
        # num_workers=8: 这是用于数据加载的子进程数量。它指定了在后台并行加载数据的工作进程数量。在这里，设置为8表示使用8个子进程加载数据，以加速数据加载。
        # pin_memory=True: 如果设置为True，则数据加载到CUDA固定内存中，这在GPU上训练时可以提高数据加载的效率。

        # Load meta-val set 加载元验证集
        self.valset = Dataset('val', self.args)  # 除了迭代器传入的numbatch为600其他不变
        self.val_sampler = CategoriesSampler(self.valset.label, 200, self.args.way,
                                             self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8,
                                     pin_memory=True)

        # Build meta-transfer learning model 构建元迁移学习模型
        self.model = MtlLearner(self.args)

        # Set optimizer 设置优化器
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
             {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler 设置学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        # load pretrained model without FC classifier 无FC分类器的负载预训练模型
        self.model_dict = self.model.state_dict()  # 首先创造一个字典
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
            print("yes")
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
                args.pre_gamma) + '_step' + \
                             str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']  # 包含了加载的预训练模型参数的字典
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)

        # # HT元批处理的超参数 后来更新
        # self.ht_meta_batch_size = args.ht_meta_batch_size  # 每个HT元批处理中的任务数量
        # self.ht_meta_lr_range = (args.ht_meta_lr_min, args.ht_meta_lr_max)  # 元学习率范围
        # self.ht_update_step_range = (args.ht_update_step_min, args.ht_update_step_max)  # 更新步骤范围

        # Set model to GPU 将模型设置为GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.保存检查点的函数。
        Args:   保存模型的权重，防止中断导致的信息丢失
          name: the name for saved checkpoint已保存检查点的名称
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the meta-train phase."""  # 元训练阶段的功能
        # 创建一个用于存储失败样本的目标文件夹
        target_folder = '/root/dataset/bili-duofenlei/HT_val'
        os.makedirs(target_folder, exist_ok=True)
        # Set the meta-train log 设置元训练日志
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
        trlog['val_HT_loss'] = []
        trlog['val_HT_acc'] = []
        trlog['val_HT_pre'] = []
        trlog['val_HT_sen'] = []
        trlog['val_HT_spec'] = []
        # trlog['train_AUC'] = []
        # trlog['val_AUC'] = []
        # trlog['train_F1'] = []
        # trlog['val_F1'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        trlog['max_pre'] = 0.0
        trlog['max_sen'] = 0.0
        trlog['max_spec'] = 0.0
        # trlog['max_AUC'] = 0.0
        # trlog['max_F1'] = 0.0
        trlog['max_pre_epoch'] = 0
        trlog['max_sen_epoch'] = 0
        trlog['max_spec_epoch'] = 0
        # trlog['max_AUC_epoch'] = 0
        # trlog['max_F1_epoch'] = 0

        # Set the timer 设置时间
        timer = Timer()
        # Set global count to zero 将全局计数设置为零
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # 初始化连续准确率不增长的计数器
        no_improvement_count = 0
        best_accuracy = 0  # 用于记录最佳准确率

        # Start meta-train 启动元训练
        for epoch in range(1, self.args.max_epoch + 1):
            failed_samples_path = []
            failed_samples_count = {}

            failed_samples_paths = []
            # Generate the labels for train set of the episodes 为剧集的训练集生成标签
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)
            # Update learning rate 更新学习率
            self.lr_scheduler.step()
            # Set the model to train mode 将模型设置为测试模式
            self.model.train()
            self.model.mode = "preval"
            # Set averager classes to record training losses and accuracies 设置平均类以记录训练损失和准确性
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_pre_averager = Averager()
            train_sen_averager = Averager()
            train_spec_averager = Averager()
            # train_AUC_averager = Averager()
            # train_F1_averager = Averager()
            train_HT_loss_averager = Averager()
            train_HT_acc_averager = Averager()
            train_HT_pre_averager = Averager()
            train_HT_sen_averager = Averager()
            train_HT_spec_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates 在元训练更新期间为剧集的测试集生成标签
            label = torch.arange(self.args.way).repeat(self.args.train_query)  #
            # print(f"label={label}")
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            # Using tqdm to read samples from train loader 用tqdm读取装载机样本
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
            if epoch > 1:
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

                    for i, batch in enumerate(self.HT_val_loader, 1):
                        # print(f'batch:{batch}')
                        if torch.cuda.is_available():
                            data, _ = [_.cuda() for _ in batch]
                        else:
                            data = batch[0]
                        p2 = self.args.way * 3
                        data_shot, data_query = data[:p2], data[p2:]
                        HT_logits = self.model((data_shot, label_shot2, data_query))
                        HT_loss = F.cross_entropy(HT_logits, label2)
                        HT_acc = count_acc(HT_logits, label2)
                        HT_pre = precision(label2, HT_logits)
                        HT_sen = sensitivity(label2, HT_logits)
                        HT_spec = specificity(label2, HT_logits)

                        predictions = HT_logits.argmax(dim=1)  # 获取预测值
                        incorrect_indices = (predictions != label2).nonzero()[:, 0]  # 找到预测错误的样本索引
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
                            'Epoch {}, HT_Loss={:.4f} HT_Acc={:.4f} HT_pre={:.4f} HT_sen={:.4f} HT_spec={:.4f}'.format(
                                epoch, HT_loss.item(), HT_acc, HT_pre, HT_sen, HT_spec))

                        # Add loss and accuracy for the averagers 增加平均值的损失和准确性
                        train_HT_loss_averager.add(HT_loss.item())
                        train_HT_acc_averager.add(HT_acc)
                        train_HT_pre_averager.add(HT_pre)
                        train_HT_sen_averager.add(HT_sen)
                        train_HT_spec_averager.add(HT_spec)
                        # train_AUC_averager.add(AUC)
                        # train_F1_averager.add(F1)

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
                # Update global count number 更新全局计数
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[
                                                  p:]  # N-way K-shot Q-query问题，N各类别，每个类别抽取K个样本用于测试，然后再n个类别，每个类别抽取K个样本用于验证
                # Output logits for model 输出模型的logits
                logits = self.model((data_shot, label_shot, data_query))

                loss = F.cross_entropy(logits, label)
                # Calculate meta-train accuracy 计算元训练精度
                acc = count_acc(logits, label)
                pre = precision(label, logits)
                sen = sensitivity(label, logits)
                spec = specificity(label, logits)
                #
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

                # Write the tensorboardX records 编写tensorboardX记录
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                writer.add_scalar('data/pre', float(pre), global_count)
                writer.add_scalar('data/sen', float(sen), global_count)
                writer.add_scalar('data/spec', float(spec), global_count)
                # writer.add_scalar('data/AUC', float(AUC), global_count)
                # writer.add_scalar('data/F1', float(F1), global_count)
                # Print loss and accuracy for this step 此步骤的打印损失和准确性
                tqdm_gen.set_description(
                    'Epoch {}, Loss={:.4f} Acc={:.4f} pre={:.4f} sen={:.4f} spec={:.4f}'.format(epoch, loss.item(), acc,
                                                                                                pre, sen, spec))

                # Add loss and accuracy for the averagers 增加平均值的损失和准确性
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                train_pre_averager.add(pre)
                train_sen_averager.add(sen)
                train_spec_averager.add(spec)
                # train_AUC_averager.add(AUC)
                # train_F1_averager.add(F1)

                # Loss backwards and optimizer updates 损失向后和优化器更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 筛选出现次数大于等于八的文件

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
            # Update the averagers 更新平均值
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            train_pre_averager = train_pre_averager.item()
            train_sen_averager = train_sen_averager.item()
            train_spec_averager = train_spec_averager.item()
            # train_AUC_averager = train_AUC_averager.item()
            # train_F1_averager = train_F1_averager.item()

            # # 找到出现次数前一百的文件
            # top_failed_samples = sorted_failed_samples[:100]
            # # print(f'top={top_failed_samples}')

            # Start validation for this epoch, set model to eval mode 开始验证此epoch，将模型设置为eval模式
            self.model.eval()
            self.model.mode = 'preval'
            # Set averager classes to record validation losses and accuracies 设置平均值类别以记录验证损失和准确性
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_pre_averager = Averager()
            val_sen_averager = Averager()
            val_spec_averager = Averager()
            # val_AUC_averager = Averager()
            # val_F1_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch 在meta-val期间为该时期的剧集测试集生成标签
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # 设置日志文件的路径
            logfile = '/root/meta-transfer-learning/logs/all_logs/filetext.log'

            # 配置日志记录
            logging.basicConfig(filename=logfile, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s: %(message)s')

            # Start meta-test 启动元测试
            if epoch > 108:
                logging.info(f'trainval: epoch{epoch}\n')

            # Print previous information 打印以前的信息
            if epoch % 10 == 0:
                print('Best Val acc={:.4f} Best Val pre={:.4f} Best Val sen={:.4f} Best Val spec={:.4f} '
                      .format(trlog['max_acc'], trlog['max_pre'], trlog['max_sen'], trlog['max_spec']))

            val_acc_list = []
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
                # val_AUC_averager.add(AUC)
                # val_F1_averager.add(F1)
                val_acc_list.append(acc)

            # Update validation averagers 更新验证平均值
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_pre_averager = val_pre_averager.item()
            val_sen_averager = val_sen_averager.item()
            val_spec_averager = val_spec_averager.item()
            # val_AUC_averager = val_AUC_averager.item()
            # val_F1_averager = val_F1_averager.item()
            # Create FailedDatasetLoader instance

            # Write the tensorboardX records 编写tensorboardX记录
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            writer.add_scalar('data/val_pre', float(val_pre_averager), epoch)
            writer.add_scalar('data/val_sen', float(val_sen_averager), epoch)
            writer.add_scalar('data/val_spec', float(val_spec_averager), epoch)
            # writer.add_scalar('data/val_AUC', float(val_AUC_averager), epoch)
            # writer.add_scalar('data/val_F1', float(val_F1_averager), epoch)
            # Print loss and accuracy for this epoch 此时期的打印损失和准确性
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

            # Update best saved model 更新最佳保存模型
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
                self.save_model('epoch' + str(epoch))

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

            # Save log 保持日志
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
            # print(f'top_failed_samples={top_failed_samples}')
            
            # Hard Task Meta-Batch 策略
            hard_task_indices = np.argsort(val_acc_list)[:3]  # 选择准确度最低的 10 个任务
            self.model.train()  # 确保模型处于训练模式

            # Adjust learning rate for hard task training
            original_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5  # Decrease learning rate by a factor of 10 (adjust as needed)

            for idx in hard_task_indices:
                hard_task_data, hard_task_label = self.val_sampler.sample_task_by_index(idx, self.valset)
                if torch.cuda.is_available():
                    hard_task_data = hard_task_data.cuda()
                    hard_task_label = hard_task_label.cuda()

                # print("hard_task_data shape:", hard_task_data.shape)
                # print("hard_task_label shape:", hard_task_label.shape)

                p = self.args.shot * self.args.way
                data_shot, data_query = hard_task_data[:p], hard_task_data[p:]

                # print("data_shot shape:", data_shot.shape)
                # print("data_query shape:", data_query.shape)

                logits = self.model((data_shot, hard_task_label[:p], data_query))
                loss = F.cross_entropy(logits, hard_task_label[p:])
                acc = count_acc(logits, hard_task_label[p:])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(f"Task {idx}: loss = {loss.item()}, acc = {acc}")
            # Restore original learning rate after hard task training
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = original_lr

        writer.close()

    def eval(self):
        failed_samples_path = []
        failed_samples_count = {}
        failed_samples_paths = []
        """The function for the meta-eval phase."""  # 元评估阶段的函数
        # Load the logs 加载日志
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))
        # trlog = torch.load('/zsm/wky/breast/meta-transfer-learning-main/pytorch/logs/meta/MiniImageNet_ResNet_MTL_shot10_way3_query10_step10_gamma0.5_lr10.0001_lr20.001_batch200_maxepoch110_baselr0.01_updatestep100_stepsize10_exp1_400X5_new/trlog')
        # Load meta-test set 加载元测试集
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 200, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder 设置测试精度记录器
        test_acc_record = np.zeros((200,))
        test_pre_record = np.zeros((200,))
        test_sen_record = np.zeros((200,))
        test_spec_record = np.zeros((200,))
        # test_AUC_record = np.zeros((600,))
        # test_F1_record = np.zeros((600,))

        # Load model for meta-test phase 元测试阶段的负载模型
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode 将模型设置为评估模式
        self.model.eval()

        # Set accuracy averager 设置精度平均值
        ave_acc = Averager()
        max_acc = Averager()
        ave_pre = Averager()
        ave_sen = Averager()
        ave_spec = Averager()
        # ave_AUC = Averager()
        # ave_F1 = Averager()

        # Generate labels 生成标签
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        # print(f'label={label}')
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

        # failed_samples = []  # 存储测试失败的样本
        # Start meta-test 启动元测试
        logging.info(f'test\n')
        max = 0

        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
            pre = precision(label, logits)
            sen = sensitivity(label, logits)
            spec = specificity(label, logits)

            predictions = logits.argmax(dim=1)  # 获取预测值
            incorrect_indices = (predictions != label).nonzero()[:, 0]  # 找到预测错误的样本索引
            # print(f'inc/r={incorrect_indices}')
            for index in incorrect_indices:
                index = int(index)
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
            ave_acc.add(acc)
            max_acc.add(acc)
            ave_pre.add(pre)
            ave_sen.add(sen)
            ave_spec.add(spec)
            # ave_AUC.add(AUC)
            # ave_F1.add(F1)
            test_acc_record[i - 1] = acc
            test_pre_record[i - 1] = pre
            test_sen_record[i - 1] = sen
            test_spec_record[i - 1] = spec
            # test_AUC_record[i - 1] = AUC
            # test_F1_record[i - 1] = F1

            if i % 40 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
                if max_acc.item() > max:
                    max = max_acc.item()
                max_acc.reset()

        # Evaluate on hard tasks
        val_acc_list = []
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = self.args.shot * self.args.way
            data_shot, data_query = data[:p], data[p:]
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
            val_acc_list.append(acc)

        hard_task_indices = np.argsort(val_acc_list)[:3]  # Select 10 tasks with lowest accuracy
        hard_task_accs = []
        for idx in hard_task_indices:
            hard_task_data, hard_task_label = self.val_sampler.sample_task_by_index(idx, test_set)
            if torch.cuda.is_available():
                hard_task_data = hard_task_data.cuda()
                hard_task_label = hard_task_label.cuda()
            p = self.args.shot * self.args.way
            data_shot, data_query = hard_task_data[:p], hard_task_data[p:]
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
            hard_task_accs.append(acc)
        
        hard_task_avg_acc = np.mean(hard_task_accs)
        print('Hard Task Average Accuracy: {:.2f}'.format(hard_task_avg_acc * 100))
        # logfile = f'/zsm/wky/breast/meta-transfer-learning-main/pytorch/logs/all_logs/{self.args.dataset}_meta_{self.args.shot}.log'
        # logging.basicConfig(filename=logfile, level=logging.INFO,
        #                     format='%(asctime)s - %(levelname)s: %(message)s')
        # Calculate the confidence interval, update the logs 计算置信区间，更新日志

        # 将 failed_samples 列表中的失败样本记录到日志中
        # logging.info(f'batch:{i},Failed samples: %s\n', failed)

        # print(sorted_failed_samples)
        logging.info('失败者:{}'.format(sorted_failed_samples))

        # 打印最大值
        print('Maximum test accuracy: {:.2f}'.format(max * 100))
        # logging.info('max Test Acc {:.4f}'.format(max_test_acc.item()))

        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'],
                                                                      ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        logging.info('Test Acc {:.4f}'.format(ave_acc.item()))
        logging.info('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print('Hard Task Acc {:.4f}'.format(hard_task_avg_acc))

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

        # m, pm = compute_confidence_interval(test_AUC_record)
        # print('Val Best Epoch {}, AUC {:.4f}, Test AUC {:.4f}'.format(trlog['max_AUC_epoch'], trlog['max_AUC'], ave_AUC.item()))
        # print('Test AUC {:.4f} + {:.4f}'.format(m, pm))
        # logging.info('Test AUC {:.4f}'.format(ave_AUC.item()))
        # logging.info('Test AUC {:.4f} + {:.4f}'.format(m, pm))
        #
        # m, pm = compute_confidence_interval(test_acc_record)
        # print('Val Best Epoch {}, F1 {:.4f}, Test F1 {:.4f}'.format(trlog['max_F1_epoch'], trlog['max_F1'], ave_F1.item()))
        # print('Test F1 {:.4f} + {:.4f}'.format(m, pm))
        # logging.info('Test F1 {:.4f}'.format(ave_F1.item()))
        # logging.info('Test F1 {:.4f} + {:.4f}'.format(m, pm))

