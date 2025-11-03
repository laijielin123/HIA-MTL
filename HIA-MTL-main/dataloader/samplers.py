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
""" Sampler for dataloader. """  # 计算机负载示例。
import torch
import numpy as np

class FailedCategoriesSampler():
    """The class to generate episodic data for failed samples"""
    def __init__(self, failed_samples, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        self.failed_samples = failed_samples
        self.num_samples = len(failed_samples)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(self.num_samples)[:self.n_cls]
            for c in classes:
                batch.append(c)
            yield batch

class CategoriesSampler():
    """The class to generate episodic data"""  # 生成偶发数据的类
    # label：一个包含样本标签的列表或数组。label
    # n_batch：生成的元学习任务（或称为 batch）的数量。600
    # n_cls：每个元学习任务中包含的类别数量。2
    # n_per：每个类别中包含的样本数量。16
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.batch = None
        self.H_batch=None

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):  # self.m_ind 包含了一个列表，其中的每个元素是一个
            # 包含特定类别图像索引的PyTorch张量。这对于生成用于元学习的小批量样本非常有用，因为您可以从每个类别中选择图像以创建训练和测试任务。
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):  # self.n_batch表示要生成episodic数据集的数量
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # 从类别索引列表 self.m_ind 中随机选择 self.n_cls 个类别。这模拟了 episodic

            # 输出每次随机挑选的数据集中的类别
            # 数据集中的类别选择。
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)  # 将 batch 列表中的任务样本索引堆叠成一个张量，然后进行转置（transpose），最后通过 reshape 进行展平，
            # 以获得一个包含所有任务样本索引的一维张量。
            self.batch = batch[-30:]  # 用于训练和验证和测试
            self.H_batch = batch[-9:]  # 用于硬任务
            # # 输出二维张量
            # print("Batch tensor shape:", batch.shape)
            # print("Batch tensor content:")
            # print(self.batch)


            yield batch  # 使用 Python 的 yield 关键字将生成的任务样本索引一维张量返回为迭代器的一个元素。

    def sample_task_by_index(self, idx, dataset):
        """根据索引采样特定任务。"""
        batch = []
        classes = torch.arange(self.n_cls)
        for c in classes:
            l = self.m_ind[c]
            pos = torch.arange(idx * self.n_per, (idx + 1) * self.n_per) % len(l)
            batch.append(l[pos])
        batch = torch.stack(batch).t().reshape(-1)
        
        # 获取实际的数据，而不仅仅是索引
        data = [dataset[i] for i in batch]
        images = torch.stack([item[0] for item in data])  # assuming item[0] is the data part
        labels = torch.tensor([item[1] for item in data])  # assuming item[1] is the label part

        return images, labels