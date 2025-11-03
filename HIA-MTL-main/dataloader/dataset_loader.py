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
""" Dataloader for all datasets. """ # 所有数据集的数据加载器
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

# 新添的HT元批处理
class FailedDatasetLoader(Dataset):
    def __init__(self, failed_samples):
        self.failed_samples = failed_samples

    def __len__(self):
        return len(self.failed_samples)

    def __getitem__(self, idx):
        sample = self.failed_samples[idx]
        file_path = sample['file_path']
        true_label = sample['true_label']

        # 加载图像
        image = Image.open(file_path).convert('RGB')

        # 数据预处理
        image_size = 224
        preprocess = transforms.Compose([
                transforms.Resize(224),  # 调整图像大小
                transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
                transforms.CenterCrop(image_size),  # 中心裁剪到指定大小
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),  # 归一化，均值标准差
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        image = preprocess(image)
        print(f'truelabel:{true_label}')
        return image, true_label


class DatasetLoader(Dataset):
    """The class to load the dataset""" # 要加载数据集的类
    # miniimagenet100个类，64，16,20用于元训练，元验证和元测试  病理279个类 179个train，45个val，55个test
    def __init__(self, setname, args, train_aug=False):
        # Set the path according to train, val and test      # 根据训练、val和测试设置路径
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val')
            label_list = os.listdir(THE_PATH)
        elif setname == 'HT_val':
            THE_PATH = osp.join(args.dataset_dir, 'HT_val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label  # 为数据和标签生成空列表
        data = []
        label = []
        self.gaussian_noise_std = 0.1  # 用你期望的值替换 your_desired_value

        # Get folders' name #获取文件夹的名称
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels 获取图像的路径和标签
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside 将数据、标签和类编号设置为可从外部访问
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # 创建标签到数据的映射字典
        if setname == 'test':
            self.label_to_data = {}
            # for i in range(len(self.data)):
            #     if label[i] not in self.label_to_data:
            #         self.label_to_data[label[i]] = []
            #     self.label_to_data[label[i]].append(data[i])
            for i in range(len(self.data)):
                self.label_to_data[i] = data[i]  # 使用数据的索引作为标签，直接映射到文件位置

        # if setname == 'val':
        #     self.label_to_data = {}
        #     # for i in range(len(self.data)):
        #     #     if label[i] not in self.label_to_data:
        #     #         self.label_to_data[label[i]] = []
        #     #     self.label_to_data[label[i]].append(data[i])
        #     for i in range(len(self.data)):
        #         self.label_to_data[i] = data[i]  # 使用数据的索引作为标签，直接映射到文件位置

        if setname == 'train':
            self.label_to_data = {}
            # for i in range(len(self.data)):
            #     if label[i] not in self.label_to_data:
            #         self.label_to_data[label[i]] = []
            #     self.label_to_data[label[i]].append(data[i])
            for i in range(len(self.data)):
                self.label_to_data[i] = data[i]  # 使用数据的索引作为标签，直接映射到文件位置

        if setname == 'HT_val':
            self.label_to_data = {}
            # for i in range(len(self.data)):
            #     if label[i] not in self.label_to_data:
            #         self.label_to_data[label[i]] = []
            #     self.label_to_data[label[i]].append(data[i])
            for i in range(len(self.data)):
                self.label_to_data[i] = data[i]  # 使用数据的索引作为标签，直接映射到文件位置

            # for index, file_path in self.label_to_data.items():
            #     print(f"索引: {index}, 文件位置: {file_path}")
        #
        # # Transformation 变化
        if train_aug:  # 原版没改 train是 test不是
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize(232),  # 调整图像大小
                transforms.RandomResizedCrop(228),  # 随机裁剪并调整大小
                transforms.CenterCrop(image_size),  # 中心裁剪到指定大小
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),  # 归一化，均值标准差
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        # # Transformation 变化  第一版改
        # if train_aug:
        #     image_size = 80
        #     self.transform = transforms.Compose([
        #         transforms.Resize(92),
        #         transforms.RandomResizedCrop(88),
        #         transforms.CenterCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        # else:
        #     image_size = 80
        #     self.transform = transforms.Compose([
        #         transforms.Resize(92),
        #         transforms.CenterCrop(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

        # if train_aug:
        #     image_size = 200
        #     self.transform = transforms.Compose([
        #         transforms.Resize(208),
        #         transforms.RandomResizedCrop(212),
        #         transforms.CenterCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        # else:
        #     image_size = 200
        #     self.transform = transforms.Compose([
        #         transforms.Resize(208),
        #         transforms.CenterCrop(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


    def __len__(self):
        # print(f"数据集的长度是{len(self.data)}")
        return len(self.data)  # 180169

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))  # 包含打开图像转变为RGB并对图像进行增强
        # print(label)

        return image, label