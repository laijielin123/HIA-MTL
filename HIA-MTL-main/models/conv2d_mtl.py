##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/pytorch/pytorch
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" MTL CONV layers. """
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

# 定义了一个自定义的卷积层，支持元迁移学习。提供了参数检查、权重和偏置的定义和初始化方法。额外描述方法用于提供层的字符串描述，便于调试
# 用于实现元迁移卷积（meta-transfer convolution）。该类继承自 torch.nn.Module，并定义了卷积层的参数和操作
class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNdMtl, self).__init__()
        # 检查输入通道数和输出通道数是否可以被组数整除，不然就抛出异常
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels       # 输入通道数
        self.out_channels = out_channels     # 输出通道数
        self.kernel_size = kernel_size       # 卷积
        self.stride = stride                 # 步长
        self.padding = padding               # 填充
        self.dilation = dilation             # 扩张
        self.transposed = transposed         # 转置
        self.output_padding = output_padding # 输出填充
        self.groups = groups                 # 组数
        # 根据是否转置卷积定义权重参数 self.weight 和元迁移权重参数 self.mtl_weight。
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        # 设置 self.weight.requires_grad = False 表示不对权重进行梯度计算。
        self.weight.requires_grad=False
        # 如果使用偏置，定义 self.bias 和 self.mtl_bias 参数，并设置 self.bias.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad=False
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        # 如果不使用偏置，注册 bias 和 mtl_bias 参数为 None。
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        # 初始化权重和偏置
        self.reset_parameters()

    # 重置参数方法
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        # 计算标准差 stdv，并初始化权重和偏置
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv) 用均匀分布初始化权重
        self.weight.data.uniform_(-stdv, stdv)
        # self.mtl_weight.data.uniform_(1, 1) 用常数1初始化元迁移权重 （缩放）
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv) 用均匀分布初始化偏置
            self.bias.data.uniform_(-stdv, stdv)
            # 和 self.mtl_bias.data.uniform_(0, 0) 用常数0初始化元迁移学习偏置 （平移）
            self.mtl_bias.data.uniform_(0, 0)

    # 额外描述方法 提供一个包含层参数的字符串描述，便于打印和调试。
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

# 用于实现二维元迁移卷积（meta-transfer convolution）
class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""
    # 初始化，参数包括输入通道数 in_channels，输出通道数 out_channels，卷积核大小 kernel_size，
    # 步幅 stride，填充 padding，扩张 dilation，组数 groups 和是否使用偏置 bias
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        # 使用 _pair 函数将参数转换为元组形式，以便适应二维卷积操作。
        # 例如，kernel_size 如果是单个整数，会被转换为 (kernel_size, kernel_size)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # 调用父类 _ConvNdMtl 的构造函数进行初始化
        # 传入的参数包括 in_channels、out_channels、kernel_size、stride、padding、
        # dilation、transposed（设置为 False）、output_padding（设置为 (0, 0)）、groups 和 bias
        super(Conv2dMtl, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 用于计算卷积层的输出
    def forward(self, inp):
        # 将 mtl_weight 扩展到与 weight 相同的形状
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        # 对权重和元迁移权重进行元素级相乘，得到新的权重 new_weight  （缩放）
        new_weight = self.weight.mul(new_mtl_weight)
        # 检查是否有偏置。如果有，则将偏置 bias 和元传输偏置 mtl_bias 相加，得到新的偏置 new_bias。否则，将 new_bias 设置为 None   （平移）
        if self.bias is not None:
            new_bias = self.bias + self.mtl_bias
        else:
            new_bias = None
        # F.conv2d 计算卷积操作。参数包括输入数据 inp、新的权重 new_weight、新的偏置 new_bias、
        # 步幅 self.stride、填充 self.padding、扩张 self.dilation 和组数 self.groups。
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)
