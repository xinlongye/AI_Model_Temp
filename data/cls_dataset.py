# -*- coding:utf-8 -*-
# @FileName  :cls_dataset.py
# @Time      :2025/10/6 18:14
# @Author    :yxl


import os

import torch
from model_utils.registry import DATASET_REGISTRY
from model_utils.tools import letter_box
import cv2
import random
import numpy as np


@DATASET_REGISTRY.register()
class ClsDataset(torch.utils.data.Dataset):
    """
    分类数据集
    """

    def __init__(self, image_size, dataset_path, augment):
        super(ClsDataset, self).__init__()

        # 读取数据集文件
        with open(dataset_path, "r", encoding='utf-8') as f:
            self.annotation_lines = f.readlines()

        # 图像输出尺寸, (w, h)
        self.image_shape = image_size

        # 是否增强
        self.augment = augment

    def __getitem__(self, index):
        img_path = self.annotation_lines[index].split(';')[0].split()[0]
        image = cv2.imread(img_path, 1)

        label = int(self.annotation_lines[index].split(';')[1].split()[0])

        # 随机增强
        if self.augment:
            image = self.generate_augment_data(image=image)

        # 检查尺寸 letter box
        if image.shape[0] != self.image_shape[1] or image.shape[1] != self.image_shape[0]:
            image = letter_box(image, self.image_shape[0], self.image_shape[1])

        # 归一化并转换为tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image / 255.0 - mean) / std
        image = np.array(image, dtype=np.float32)
        image = image.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(image)

        return input_tensor, label

    def __len__(self):
        return len(self.annotation_lines)

    def equivalent_preprocess(self, num_choose, save_dir):
        total_sample = len(self.annotation_lines)
        if num_choose > total_sample:
            num_choose = total_sample

        interval = total_sample / num_choose
        inference_data = []
        visual_images = []
        for i in range(num_choose):
            index = int(round(i * interval))
            img_path = self.annotation_lines[index].split(';')[0].split()[0]
            image = cv2.imread(img_path, 1)

            label = int(self.annotation_lines[index].split(';')[1].split()[0])

            # 检查尺寸 letter box
            if image.shape[0] != self.image_shape[1] or image.shape[1] != self.image_shape[0]:
                image = letter_box(image, self.image_shape[0], self.image_shape[1])
            visual_images.append(image)

            # 保存输入
            save_img_path = os.path.join(save_dir, f"{i}_aim_img_{label}.jpg")
            if not os.path.exists(save_img_path):
                cv2.imwrite(save_img_path, image)

            # 归一化并转换为tensor
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.float32)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = (image / 255.0 - mean) / std
            image = np.array(image, dtype=np.float32)
            image = np.expand_dims(image.transpose(2, 0, 1), 0)
            input_tensor = torch.from_numpy(image)

            inference_data.append(input_tensor)
        return inference_data, visual_images

    def generate_augment_data(self, image):
        # 随机水平翻转 (50%概率)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # 1表示水平翻转

        # 随机垂直翻转 (20%概率)
        if random.random() < 0.2:
            image = cv2.flip(image, 0)  # 0表示垂直翻转

        # 随机旋转（0, 90, 180, 270度）
        random_angle = random.choice([0, 90, 180, 270])
        if random_angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif random_angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif random_angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 随机亮度/对比度调整
        if random.random() < 0.3:
            # 转换为HSV便于调整亮度
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 亮度调整 (-20% 到 +20%)
            value = hsv[:, :, 2]
            value = value * (0.8 + 0.4 * random.random())  # 0.8-1.2倍
            value = np.clip(value, 0, 255).astype(np.uint8)
            hsv[:, :, 2] = value
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 随机裁剪（保留中心区域70%-100%）
        if random.random() < 0.3:
            h, w = image.shape[:2]
            scale = 0.7 + 0.3 * random.random()  # 0.7-1.0
            new_h, new_w = int(h * scale), int(w * scale)
            # 随机选择裁剪区域
            y = random.randint(0, h - new_h)
            x = random.randint(0, w - new_w)
            image = image[y:y + new_h, x:x + new_w]
            # 裁剪后resize回原尺寸
            image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_AREA)

        return image
