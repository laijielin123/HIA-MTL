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
""" Additional utility functions. """  # 附加实用程序功能。
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F


def ensure_path(path):
    """The function to make log path. 生成日志路径的函数。
    Args:该函数的主要作用是确保指定的路径在文件系统中存在，以便可以安全地将文件保存到该路径下。如果路径已存在，
    函数不会做任何改变，如果路径不存在，它将创建这个路径。这在许多情况下很有用，特别是在保存实验结果、日志文件或模型文件时。
      path: the generated saving path.生成的保存路径。
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


class Averager():
    """The class to calculate the average."""  # 要计算平均值的类， n表示当前的样本数，v表示当前的病均值

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):   # 添加归零
        self.n = 0
        self.v = 0

def count_acc(logits, label):
    """The function to calculate the .要计算的函数。
    Args:
      logits: input logits.输入logits。通常是一个形状为 (batch_size, num_classes) 的张量，其中 batch_size 表示批次大小，
      num_classes 表示分类的类别数。
      label: ground truth labels. 实际的标签，通常是一个形状为 (batch_size,) 的张量，包含每个样本的真实类别标签。
    Return:
      The output accuracy. 输出精度。
      首先，它使用 F.softmax(logits, dim=1) 对 logits 进行 softmax 操作，以获得每个类别的概率分布。
然后，使用 .argmax(dim=1) 找到每个样本中概率最高的类别的索引，即模型预测的类别。
接下来，函数计算模型的预测结果与实际标签相等的样本数量，并将其除以总的样本数量，从而计算得到准确度。
最后，函数返回准确度作为输出
    """
    # pred = F.softmax(logits, dim=1).argmax(dim=1)
    # if torch.cuda.is_available():
    #     return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    # return (pred == label).type(torch.FloatTensor).mean().item()

    # 获取模型预测的类别
    predicted_classes = torch.argmax(logits, dim=1)

    # 计算预测正确的数量
    correct_predictions = (predicted_classes == label).sum().item()

    # 计算准确率
    accuracy = correct_predictions / label.size(0)

    return accuracy


def precision(label, logits):  # 写一个精确度Pre,其中label是真实标签，logits是模型的预测值
    # 计算精确度
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    true_positives = sum((label == 1) & (pred == 1))
    false_positives = sum((label == 0) & (pred == 1))

    if (true_positives + false_positives) == 0:
        return 0.0
    else:
        precision_score = true_positives / (true_positives + false_positives)
        return precision_score


def sensitivity(label, logits):  # 写一个计算灵敏度/召回率
    # 计算灵敏度
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    true_positives = ((pred == 1) & (label == 1)).sum().item()
    false_negatives = ((pred == 0) & (label == 1)).sum().item()

    if (true_positives + false_negatives) == 0:
        return 0.0
    else:
        sensitivity_score = true_positives / (true_positives + false_negatives)
        return sensitivity_score


def specificity(label, logits):  # 计算特异性spec
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    true_negatives = sum((label == 0) & (pred == 0))
    false_positives = sum((label == 0) & (pred == 1))

    if (true_negatives + false_positives) == 0:
        return 0.0
    else:
        specificity = true_negatives / (true_negatives + false_positives)
        return specificity


class Timer():
    """The class for timer."""  # 计时器的类__init__: 初始化计时器，记录当前时间。measure(p=1): 这个方法接受一个可选参数 p，

    # 它表示时间的分毫秒（p=1表示以秒为单位，默认值）。方法计算自计时器初始化以来经过的时间，并返回一个格式化的字符串，表示经过的时间。
    # 如果经过的时间大于等于1小时，它将以小时为单位显示，如果大于等于1分钟，以分钟为单位显示，否则以秒为单位显示。
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """The function to calculate the .要计算的函数。
    Args:
      data: input records数据：输入记录
      label: ground truth labels.标签：基本事实标签。
    Return:
      m: mean value平均值
      pm: confidence interval.置信区间。
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)  # 平均值
    std = np.std(a)  # 标准差
    pm = 1.96 * (std / np.sqrt(len(a)))  # 采用了 1.96 的标准分数，这对应于 95% 的置信水平。置信区间表示均值 m 在给定置信水平下的上下限范围。
    return m, pm
