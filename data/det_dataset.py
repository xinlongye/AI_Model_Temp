# -*- coding:utf-8 -*-
# @FileName  :det_dataset.py
# @Time      :2025/10/28 21:58
# @Author    :yxl

import random

import cv2
import numpy as np
import torch
import yaml

from model_utils.registry import DATASET_REGISTRY


def mosaic_enhancement(label_lines, out_shape, color=(128, 128, 128)):
    """
    :param label_lines: 图像路径与标签列表
    :param out_shape: 输出尺寸 wh
    :param color: 边缘填充色彩
    :return: mosaic处理后的图像和边界框
    """
    # 计算裁剪点偏移比例
    min_offset_x = random.uniform(0.3, 0.7)
    min_offset_y = random.uniform(0.3, 0.7)
    x_cut = int(out_shape[0] * min_offset_x)
    y_cut = int(out_shape[1] * min_offset_y)

    merge_image = np.zeros(shape=(out_shape[1], out_shape[0], 3), dtype=np.uint8)

    lb_boxes_list = []
    merge_bbox = []
    # 获取图像与对应标签列表
    for index, line in enumerate(label_lines):
        # 图像数据
        img_path = line.split(';')[0].split()[0]
        image = cv2.imread(img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 标签数据
        bbox = []
        if len(line.split(';')) > 1:
            box_list = line.split(';')[1:-1]
            for box_str in box_list:
                bbox.append(np.array([float(i) for i in box_str.split(",")]))
        bbox = np.array(bbox, dtype=np.int32)

        # letterbox
        lb_image, lb_box = letterbox_ratio(src_image=image, src_box=bbox, out_shape=out_shape, place=index, color=color)

        # 添加单张图像的边界框到列表中
        lb_boxes_list.append(lb_box)

        # 添加到输出图像中
        if index == 0:
            merge_image[:y_cut, :x_cut, :] = lb_image[:y_cut, :x_cut, :]
        if index == 1:
            merge_image[:y_cut, x_cut:, :] = lb_image[:y_cut, x_cut:, :]
        if index == 2:
            merge_image[y_cut:, :x_cut, :] = lb_image[y_cut:, :x_cut, :]
        if index == 3:
            merge_image[y_cut:, x_cut:, :] = lb_image[y_cut:, x_cut:, :]

    # 处理合并所有的边界框
    for box_index, m_boxes in enumerate(lb_boxes_list):
        if len(m_boxes) == 0:
            continue
        # 遍历单张图像的边界框
        for m_box in m_boxes:
            tmp_box = []
            x1, y1, x2, y2 = m_box[0], m_box[1], m_box[2], m_box[3]
            if box_index == 0:
                if y1 > y_cut or x1 > x_cut:
                    continue
                if y2 >= y_cut >= y1:
                    y2 = y_cut
                    if y2 - y1 < 5:
                        continue
                if x2 >= x_cut >= x1:
                    x2 = x_cut
                    if x2 - x1 < 5:
                        continue

            if box_index == 1:
                if y1 > y_cut or x2 < x_cut:
                    continue
                if y2 >= y_cut >= y1:
                    y2 = y_cut
                    if y2 - y1 < 5:
                        continue
                if x2 >= x_cut >= x1:
                    x1 = x_cut
                    if x2 - x1 < 5:
                        continue

            if box_index == 2:
                if y2 < y_cut or x1 > x_cut:
                    continue
                if y2 >= y_cut >= y1:
                    y1 = y_cut
                    if y2 - y1 < 5:
                        continue
                if x2 >= x_cut >= x1:
                    x2 = x_cut
                    if x2 - x1 < 5:
                        continue

            if box_index == 3:
                if y2 < y_cut or x2 < x_cut:
                    continue
                if y2 >= y_cut >= y1:
                    y1 = y_cut
                    if y2 - y1 < 5:
                        continue
                if x2 >= x_cut >= x1:
                    x1 = x_cut
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(m_box[-1])
            merge_bbox.append(tmp_box)

    return merge_image, merge_bbox


def letterbox_ratio(src_image, src_box, out_shape, place, color=(128, 128, 128)):
    """
    :param src_image: 原图
    :param src_box: 边界框标签，x1y1x2y2
    :param out_shape: 输出尺寸 wh
    :param place: 放置位置, index==0: 左上图像;index==1: 右上图像; index==2: 左下图像; index==3: 右下图像
    :param color: 边缘填充色彩
    :return: 处理后的图像和边界框
    """
    scale_factor = random.uniform(0.7, 0.9)

    img_h = src_image.shape[0]
    img_w = src_image.shape[1]
    target_w = out_shape[0]
    target_h = out_shape[1]

    aspect_ratio = img_w / img_h  # 宽高比

    if aspect_ratio < 1:
        new_h = int(target_h * scale_factor)
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = int(target_h * scale_factor)
        new_h = int(new_w / aspect_ratio)

    scale = new_w / img_w  # 缩放比例

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    # image
    resized_img = cv2.resize(src_image, (new_w, new_h))
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if place == 0:  # 左上图像
        pad_right = pad_w
        pad_bottom = pad_h
    if place == 1:  # 右上图像
        pad_left = pad_w
        pad_bottom = pad_h
    if place == 2:  # 左下图像
        pad_right = pad_w
        pad_top = pad_h
    if place == 3:  # 右下图像
        pad_left = pad_w
        pad_top = pad_h
    letterboxed_img = cv2.copyMakeBorder(resized_img,
                                         pad_top, pad_bottom, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=color)
    if len(src_box) > 0:
        src_box = np.array(src_box, dtype=np.float32)
        scaled_box = src_box * scale

        offset = np.array([[pad_left, pad_top, pad_left, pad_top, 0]], dtype=np.float32)
        letterboxed_box = scaled_box + offset
        # check
        letterboxed_box[:, 0] = np.clip(letterboxed_box[:, 0], 0, target_w - 1)  # x1
        letterboxed_box[:, 1] = np.clip(letterboxed_box[:, 1], 0, target_h - 1)  # y1
        letterboxed_box[:, 2] = np.clip(letterboxed_box[:, 2], 0, target_w - 1)  # x2
        letterboxed_box[:, 3] = np.clip(letterboxed_box[:, 3], 0, target_h - 1)  # y2
        letterboxed_box = letterboxed_box.astype(np.int32)
    else:
        letterboxed_box = src_box

    return letterboxed_img, letterboxed_box


def letterbox(src_image, src_box, out_shape, color=(128, 128, 128)):
    """
    :param src_image: 原图
    :param src_box: 边界框标签，x1y1x2y2
    :param out_shape: 输出尺寸 wh
    :param color: 边缘填充色彩
    :return: 处理后的图像和边界框
    """
    # 计算尺寸
    img_h = src_image.shape[0]
    img_w = src_image.shape[1]

    target_w = out_shape[0]
    target_h = out_shape[1]

    scale = min(target_w / img_w, target_h / img_h)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # image
    resized_img = cv2.resize(src_image, (new_w, new_h))
    letterboxed_img = cv2.copyMakeBorder(resized_img,
                                         pad_top, pad_bottom, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=color)
    # box label
    if len(src_box) > 0:
        np.random.shuffle(src_box)
        src_box = np.array(src_box, dtype=np.float32)
        scaled_box = src_box * scale

        offset = np.array([[pad_left, pad_top, pad_left, pad_top, 0]], dtype=np.float32)
        letterboxed_box = scaled_box + offset
        # check
        letterboxed_box[:, 0] = np.clip(letterboxed_box[:, 0], 0, target_w - 1)  # x1
        letterboxed_box[:, 1] = np.clip(letterboxed_box[:, 1], 0, target_h - 1)  # y1
        letterboxed_box[:, 2] = np.clip(letterboxed_box[:, 2], 0, target_w - 1)  # x2
        letterboxed_box[:, 3] = np.clip(letterboxed_box[:, 3], 0, target_h - 1)  # y2
        letterboxed_box = letterboxed_box.astype(np.int32)
    else:
        letterboxed_box = src_box

    return letterboxed_img, letterboxed_box

def geometric_enhancement(src_image, src_box, rotate=90):
    """
    基础几何增强，一般情况下图像和边界框都会变化
    包括：平移、缩放、翻转、旋转、
    :param src_image: 原图
    :param src_box: 边界框标签，x1y1x2y2 cls
    :param rotate: 旋转角度
    :return: 几何变换后图像与边界框
    """
    # 旋转
    h, w = src_image.shape[:2]
    x1, y1, x2, y2 = src_box[:4]
    new_x1 = 0
    new_y1 = 0
    new_x2 = 0
    new_y2 = 0
    match rotate:
        case 90:
            src_image = cv2.rotate(src_image, cv2.ROTATE_90_CLOCKWISE)
            new_x1 = y1
            new_y1 = h - 1 - x2
            new_x2 = y2
            new_y2 = h - 1 - x1
        case 180:
            src_image = cv2.rotate(src_image, cv2.ROTATE_180)
            new_x1 = w - 1 - x2
            new_y1 = h - 1 - y2
            new_x2 = w - 1 - x1
            new_y2 = h - 1 - y1
        case 270:
            src_image = cv2.rotate(src_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_x1 = w - 1 - y2
            new_y1 = x1
            new_x2 = w - 1 - y1
            new_y2 = x2
        case _:
            pass
    src_box = [new_x1, new_y1, new_x2, new_y2, src_box[4]]

    return src_image, src_box


@DATASET_REGISTRY.register()
class DetDataset(torch.utils.data.Dataset):
    """
    分类数据集
    """

    def __init__(self, image_size, dataset_path, augment, epoch):
        super(DetDataset, self).__init__()

        # 读取数据集文件
        with open(dataset_path, "r", encoding='utf-8') as f:
            self.annotation_lines = f.readlines()

        # 图像输出尺寸, (w, h)
        self.image_shape = image_size

        # 是否增强(普通增强：旋转、缩放、平移、剪切、透视、上下翻转、左右翻转、通道交换等几何变换)
        # 几何变换 ： 旋转、缩放、平移、剪切、透视、上下翻转、左右翻转
        # 颜色空间增强 ： 色调调整、饱和度调整、色相调整
        # 特殊增强：Mosaic、Mixup、CutMix
        self.augment = augment

        # 总共epoch数
        self.total_epoch = epoch

        # 当前epoch数
        self.current_epoch = -1

    def __getitem__(self, index):
        """
        单张图像
        """
        # # 图像数据
        # img_path = self.annotation_lines[index].split(';')[0].split()[0]
        # image = cv2.imread(img_path, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # 标签数据
        # bbox = []
        # if len(self.annotation_lines[index].split(';')) > 1:
        #     box_list = self.annotation_lines[index].split(';')[1:-1]
        #     for box_str in box_list:
        #         bbox.append(np.array([float(i) for i in box_str.split(",")]))
        # bbox = np.array(bbox, dtype=np.int32)
        # # letterbox增强
        # image, bbox = letterbox(image, bbox, out_shape=self.image_shape)

        """
        多张图像混合增强
        """
        # mosaic: 随机挑选3张 + 当前索引图像 一共四张进行融合
        label_lines = random.sample(self.annotation_lines, 3)
        label_lines.append(self.annotation_lines[index])
        random.shuffle(label_lines)
        image, bbox = mosaic_enhancement(label_lines=label_lines, out_shape=self.image_shape, color=(128, 128, 128))

        # 在前面的epoch中进行数据增强， 最后n个epoch中不进行数据增强
        return image, bbox

    def __len__(self):
        return len(self.annotation_lines)


if __name__ == "__main__":
    print(__file__)
    opt_path = r"D:\Code\Projects\python\AI_Model_Temp\options\det_yolov8_voc_option.yml"
    with open(opt_path, "r", encoding="utf-8") as f:
        option_data = yaml.load(f, Loader=yaml.FullLoader)

    dataset_option = option_data["dataset"]

    train_param = dataset_option["train"]
    det_dataset = DetDataset(image_size=train_param["image_size"],
                             dataset_path=train_param["dataset_path"],
                             augment=train_param["augment"],
                             epoch=train_param["epoch"])   #13700
    # show
    show_image, box = det_dataset[1]

    # draw box
    for box_rect in box:
        cv2.rectangle(show_image, (box_rect[0], box_rect[1]), (box_rect[2], box_rect[3]), (0, 255, 0), 2)
    cv2.imshow("show_image", show_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
