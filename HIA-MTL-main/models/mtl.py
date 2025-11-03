##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl

# 用于实现元学习算法中的内部循环，快速适应新任务
class BaseLearner(nn.Module):
    """The class for inner loop."""
    # args 包含模型的超参数，z_dim 是输入特征的维度
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        # 用于存储模型的参数
        self.vars = nn.ParameterList()
        # self.fc1_w 是一个可学习的参数，初始化为 self.args.way 行 z_dim 列的全 1 矩阵，用于表示第一层的权重矩阵
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        # 使用 torch.nn.init.kaiming_normal_ 方法对 self.fc1_w 进行 Kaiming 正态分布初始化
        torch.nn.init.kaiming_normal_(self.fc1_w)
        # 将fc1_w权重添加到vars列表中
        self.vars.append(self.fc1_w)
        # self.fc1_b 是一个可学习的参数，初始化为 self.args.way 维的全零向量，用于表示第一层的偏置
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        # 将fc1_b偏置添加到vars列表中
        self.vars.append(self.fc1_b)

    # input_x 是输入特征，the_vars 是用于前向传播的参数
    def forward(self, input_x, the_vars=None):
      # 如果 the_vars 为 None，则使用 self.vars
        if the_vars is None:
            the_vars = self.vars
        # 从 the_vars 中提取第一层的权重 fc1_w 和偏置 fc1_b
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        # 进行线性变换（全连接层），计算输出 net
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    # 重载了 nn.Module 的 parameters 方法，使其返回 self.vars 而不是默认的参数集合
    def parameters(self):
        return self.vars

# 用于实现元学习算法中的外部循环（outer loop）。该类继承自 nn.Module，并结合了预训练和元训练的能力
class MtlLearner(nn.Module):
    """The class for outer loop."""
    # args 包含模型的超参数，mode 是模型的模式，num_cls 是类别数量
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        # self.update_lr 和 self.update_step 分别保存更新的学习率和更新步骤数
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        # z_dim 设置为 640，表示特征向量的维度
        z_dim = 640
        self.base_learner = BaseLearner(args, z_dim)

        # 根据 mode 的值，选择不同的编码器 self.encoder
        if self.mode == 'meta':
            self.encoder = ResNetMtl()  
        # 如果模式不是 'meta'，则使用 ResNetMtl(mtl=False) 并定义一个前馈网络 self.pre_fc 进行分类
        else:
            self.encoder = ResNetMtl(mtl=False)  
            self.pre_fc = nn.Sequential(nn.Linear(640, 1000), nn.ReLU(), nn.Linear(1000, num_cls))

    # 定义了前向传播过程，根据不同的模式选择不同的前向传播路径
    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    # 预训练前向传播函数
    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        # 传入的输入图像经过编码器 self.encoder 和前馈网络 self.pre_fc 进行前向传播
        return self.pre_fc(self.encoder(inp))

    # 元训练前向传播函数
    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        # 通过编码器获取 data_shot 和 data_query 的嵌入向量
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        # 使用 base_learner 计算 data_shot 的 logits，并计算交叉熵损失
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        # 通过反向传播计算梯度，并更新权重 fast_weights
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        # 使用更新后的权重计算 data_query 的 logits
        logits_q = self.base_learner(embedding_query, fast_weights)

        # 在指定的更新步骤数 self.update_step 内，重复上述更新权重和计算 logits 的过程
        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    # 预训练验证前向传播函数
    # 在预训练阶段进行元验证的前向传播过程
    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        # 通过编码器获取 data_shot 和 data_query 的嵌入向量
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        # 使用 base_learner 计算 data_shot 的 logits，并计算交叉熵损失
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        # 通过反向传播计算梯度，并更新权重 fast_weights，学习率固定为 0.01
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        # 使用更新后的权重计算 data_query 的 logits
        logits_q = self.base_learner(embedding_query, fast_weights)

        # 在固定的 100 个更新步骤内，重复上述更新权重和计算 logits 的过程
        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        # 返回data_query的最终预测结果 logits_q
        return logits_q
        
