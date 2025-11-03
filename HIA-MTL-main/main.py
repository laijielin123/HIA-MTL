##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta2 import MetaTrainer
from trainer.pre import PreTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters 基础参数
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture 网络结构
    parser.add_argument('--dataset', type=str, default='overlap-8', choices=['miniImageNet', 'MiniImageNet', 'FC100', 'random50', 'overlap', 'no-overlap']) # Dataset 数据集
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase 阶段
    parser.add_argument('--seed', type=int, default=66) # Manual seed for PyTorch, "0" means using random seed 随机数种子
    parser.add_argument('--gpu', default='1') # GPU id 
    parser.add_argument('--dataset_dir', type=str, default='/root/autodl-tmp/wang-BreakHis/8fen/40X8/') # Dataset folder 数据集路径

    # Parameters for meta-train phase 元训练阶段参数
    parser.add_argument('--max_epoch', type=int, default=100) # Epoch number for meta-train phase 元训练的最大训练轮数
    parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train 元训练中任务的数量
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task 每类样本数量
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task 每个任务中的类数量
    parser.add_argument('--train_query', type=int, default=10) # The number of training samples for each class in a task 每个类的训练样本数量
    parser.add_argument('--val_query', type=int, default=10) # The number of test samples for each class in a task 每个类的验证样本数量
    parser.add_argument('--meta_lr1', type=float, default=0.0001) # Learning rate for SS weights SS权重的学习率
    parser.add_argument('--meta_lr2', type=float, default=0.001) # Learning rate for FC weights  FC权重的学习率
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop  内循环的学习率
    parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop    内循环的更新次数
    parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates  降低元学习率的时期数量
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay       元训练学习率衰减的Gamma
    parser.add_argument('--init_weights', type=str, default='/root/meta-transfer-learning/logs/pre/overlap-8_ResNet_batchsize128_lr0.1_gamma0.2_step30_maxepoch110/max_acc.pth') # The pre-trained weights for meta-train phase  元训练阶段的预训练权重
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase  元评估阶段的元训练权重
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train              元训练的附加标签

    # Parameters for pretain phase  预训练阶段参数
    parser.add_argument('--pre_max_epoch', type=int, default=110) # Epoch number for pre-train phase   预训练的最大训练轮数
    parser.add_argument('--pre_batch_size', type=int, default=128) # Batch size for pre-train phase    预训练的批量大小
    parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase    预训练的学习率
    parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay   预训练的学习率衰减参数
    parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate   学习率衰减的步长
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train             优化器的动量
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train  优化器的权重衰减
    print(torch.cuda.is_available())
    # Set and print the parameters 解析和打印参数
    args = parser.parse_args()
    pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch  设置PyTorch随机种子
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    # 启动不同的训练或评估阶段(元训练阶段、元评估阶段、预训练阶段)
    if args.phase=='meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase=='meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    elif args.phase=='pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    else:
        raise ValueError('Please set correct phase.')
