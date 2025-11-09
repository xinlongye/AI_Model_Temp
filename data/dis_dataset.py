# -*- coding:utf-8 -*-
# @FileName  :dis_dataset.py
# @Time      :2025/10/6 16:03
# @Author    :yxl


import os

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from model_utils.registry import DATASET_REGISTRY
import cv2
import random
import numpy as np


@DATASET_REGISTRY.register()
class DISDataset(torch.utils.data.Dataset):
    """
    高精度二值分割，数据集
    """

    def __init__(self, image_size, dataset_path, augment):
        super(DISDataset, self).__init__()

        # 读取数据集文件
        with open(dataset_path, "r", encoding='utf-8') as f:
            self.annotation_lines = f.readlines()

        # 图像输出尺寸
        self.image_shape = image_size

        # 是否增强
        self.augment = augment

    def __getitem__(self, index):
        img_path = self.annotation_lines[index].split(';')[0].split()[0]
        image = cv2.imread(img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.annotation_lines[index].split(';')[1].split()[0]
        mask = cv2.imread(mask_path, 0)

        # 检查尺寸
        if image.shape[0] != self.image_shape[1] or image.shape[1] != self.image_shape[0]:
            image = cv2.resize(image, dsize=self.image_shape, interpolation=cv2.INTER_LINEAR)  # 0~255
            mask = cv2.resize(mask, dsize=self.image_shape, interpolation=cv2.INTER_NEAREST)  # 0~255

        res, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # 0 || 255

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # 归一化
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if self.augment:
            random_angle = random.choice([0, 90, 180, 270])
            image = F.rotate(image, random_angle, InterpolationMode.BILINEAR)
            mask = F.rotate(mask, random_angle, InterpolationMode.NEAREST)

        return image, mask

    def __len__(self):
        return len(self.annotation_lines)

    def equivalent_preprocess(self, num_choose, save_dir):
        total_sample = len(self.annotation_lines)
        if num_choose > total_sample:
            num_choose = total_sample

        interval = total_sample / num_choose
        visual_images = []
        for i in range(num_choose):
            index = int(round(i * interval))
            img_path = self.annotation_lines[index].split(';')[0].split()[0]
            image = cv2.imread(img_path, 1)

            mask_path = self.annotation_lines[index].split(';')[1].split()[0]
            mask = cv2.imread(mask_path, 0)

            # 检查尺寸
            if image.shape[0] != self.image_shape[1] or image.shape[1] != self.image_shape[0]:
                image = cv2.resize(image, dsize=self.image_shape, interpolation=cv2.INTER_LINEAR)  # 0~255
                mask = cv2.resize(mask, dsize=self.image_shape, interpolation=cv2.INTER_NEAREST)  # 0~255

            # 保存输入
            save_img_path = os.path.join(save_dir, f"{i}_aim_img.jpg")
            save_mask_path = os.path.join(save_dir, f"{i}_aim_mask.png")
            if not os.path.exists(save_img_path):
                cv2.imwrite(save_img_path, image)
            if not os.path.exists(save_mask_path):
                cv2.imwrite(save_mask_path, mask)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 归一化
            image = np.array(image, dtype=np.float32)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = (image / 255.0 - mean) / std
            image = np.array(image, dtype=np.float32)

            image = np.expand_dims(image.transpose(2, 0, 1), 0)
            input_tensor = torch.from_numpy(image)

            visual_images.append(input_tensor)
        return visual_images