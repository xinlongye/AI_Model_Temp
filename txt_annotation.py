# -*- coding:utf-8 -*-
# @FileName  :txt_annotation.py
# @Time      :2025/10/6 18:11
# @Author    :yxl

import glob
import json
import os

import numpy as np
from tqdm import tqdm
import xml

from model_utils.tools import get_classes


def generate_animal_data_txt():
    print("生成动物数据对应的标签文件")
    """
    参数设置
    """
    classes_txt_path = r"/Dataset/animals_90/cls_classes.txt"
    save_dir = r"/AI_Model_Temp/dataset/animal_cls"
    dataset_dir = r'/Dataset/animals_90'
    datasets = ["train", "valid", "test"]

    """
    划分数据集,数据集的组织格式如下：
    --datasets_path
        --train
            --animal_0
                --1.jpg
                --2.jpg
            --animal_1
                --1.jpg
                --2.jpg
                ...
        --valid
            ...
        --test
            ...
    """
    class_name_list, len_class = get_classes(classes_txt_path)
    print(f"sum_class = {len_class}, classes = {class_name_list}")

    num_image = 0
    for dataset_name in datasets:
        save_txt_path = os.path.join(save_dir, "animal_cls_" + dataset_name + ".txt")
        annotation_file = open(save_txt_path, "w", encoding='utf-8')

        # 遍历目录
        dataset_path = os.path.join(dataset_dir, dataset_name)
        for class_name in tqdm(os.listdir(dataset_path)):
            if class_name not in class_name_list:
                continue

            cls_id = class_name_list.index(class_name)

            image_path_list = glob.glob(os.path.join(dataset_path, class_name, "*.jpg"))
            for image_path in image_path_list:
                annotation_file.write(f"{image_path};{cls_id};")
                annotation_file.write("\n")
                num_image += 1
        annotation_file.close()

    print("num_dataset = ", num_image)


def generate_cat_dog_data_txt():
    print("生成猫狗数据对应的标签文件")
    """
    参数设置
    """
    classes_txt_path = r"cat_dog/datasets/cls_classes.txt"
    save_dir = r"/dataset/cat_dog_cls"
    dataset_dir = r'/cat_dog/datasets/'

    """
    划分数据集,数据集的组织格式如下：
    --datasets_path
        --train
            --cat
                --1.jpg
                --2.jpg
            --dog
                --1.jpg
                --2.jpg
                ...
        --test
            ...
    """
    class_name_list, len_class = get_classes(classes_txt_path)
    print(f"sum_class = {len_class}, classes = {class_name_list}")

    # 测试集
    save_test_txt_path = os.path.join(save_dir, "cat_dog_cls_test.txt")
    test_annotation_file = open(save_test_txt_path, "w", encoding='utf-8')

    num_test = 0
    test_dataset_path = os.path.join(dataset_dir, "test")
    file_list = os.listdir(test_dataset_path)
    for i in file_list:
        if i not in class_name_list:
            continue
        if i in class_name_list:
            cls_id = class_name_list.index(i)
            image_path_list = glob.glob(os.path.join(test_dataset_path, i, "*.jpg"))
            for j in tqdm(image_path_list):
                test_annotation_file.write(f"{j};{cls_id};")
                test_annotation_file.write("\n")
                num_test += 1
    test_annotation_file.close()
    print("测试集数量 = ", num_test)

    # 训练集
    train_percent = 0.9  # 训练集占比
    save_train_txt_path = os.path.join(save_dir, "cat_dog_cls_train.txt")
    save_val_txt_path = os.path.join(save_dir, "cat_dog_cls_val.txt")

    train_annotation_file = open(save_train_txt_path, "w", encoding='utf-8')
    val_annotation_file = open(save_val_txt_path, "w", encoding='utf-8')

    num_train_total = 0
    num_val_total = 0
    train_val_dataset_path = os.path.join(dataset_dir, "train")
    for i in file_list:
        if i not in class_name_list:
            continue
        if i in class_name_list:
            cls_id = class_name_list.index(i)
            image_path_list = glob.glob(os.path.join(train_val_dataset_path, i, "*.jpg"))
            num_train_val = len(image_path_list)
            num_train = int(num_train_val * train_percent)  # 训练集个数

            # 训练集
            train_file_list = image_path_list[: num_train]
            for j in tqdm(train_file_list):
                train_annotation_file.write(f"{j};{cls_id};")
                train_annotation_file.write("\n")
                num_train_total += 1

            # 验证集
            val_file_list = image_path_list[num_train:]
            for j in tqdm(val_file_list):
                val_annotation_file.write(f"{j};{cls_id};")
                val_annotation_file.write("\n")
                num_val_total += 1

    train_annotation_file.close()
    val_annotation_file.close()
    print("训练集数量 = ", num_train_total)
    print("验证集数量 = ", num_val_total)


def generate_sky_data_txt():
    """
    参数设置
    """
    datasets_path = r"/sky_19k_320/"
    save_dir = r"/sky_seg"
    datasets = ["train", "val", "test"]
    img_labels = ["img", "mask"]
    train_val_percent = 0.9  # 训练验证集的占比
    train_percent = 0.9  # 训练集占比

    """
    划分数据集,数据集的组织格式如下：
    --datasets_path
        --img
            --1.jpg
            --2.jpg
            ...
        --mask
            --1.png
            --2.png
            ...
    """
    img_name_list = os.listdir(os.path.join(datasets_path, img_labels[0]))
    np.random.shuffle(img_name_list)  # 打乱
    num_dataset = len(img_name_list)
    num_train_val = int(num_dataset * train_val_percent)
    num_train = int(num_train_val * train_percent)  # 训练集个数
    num_val = num_train_val - num_train  # 验证集个数
    num_test = num_dataset - num_train_val  # 测试集个数

    print("数据集图像数量 = ", num_dataset)
    print("训练集图像数量 = %d (%.2f)" % (num_train, num_train / num_dataset))
    print("验证集图像数量 = %d (%.2f)" % (num_val, num_val / num_dataset))
    print("测试图像数量 = %d (%.2f)" % (num_test, num_test / num_dataset))

    """
    保存标签
    """

    for dataset in datasets:
        file_name_list = None
        if dataset == datasets[0]:
            # 训练集
            file_name_list = img_name_list[: num_train]
        elif dataset == datasets[1]:
            # 验证集
            file_name_list = img_name_list[num_train: num_train + num_val]
        else:
            # 训练集
            file_name_list = img_name_list[num_train + num_val:]

        save_path = os.path.join(save_dir, 'sky_seg_' + dataset + '.txt')
        list_file = open(save_path, 'w')
        for file_name in tqdm(file_name_list):
            # 检查文件后缀
            if os.path.splitext(file_name)[-1] != ".jpg":
                continue
            img_path = os.path.join(datasets_path, img_labels[0], file_name)
            mask_path = os.path.join(datasets_path, img_labels[1], os.path.splitext(file_name)[0] + ".png")
            list_file.write("%s" % img_path + ";" + '%s' % mask_path + ";")
            list_file.write('\n')
        list_file.close()


def generate_duts_data_txt():
    """
    参数设置
    """
    file_name_list = None
    dataset_dir = r"I:\Dataset\DUTS\OpenDataLab___DUTS\raw"
    save_dir = r"D:\Code\Projects\python\AI_Model_Temp\data_txt\duts_seg"

    data_category = ["DUTS-TR", "DUTS-TE"]
    img_labels = ["Image", "Mask"]
    datasets = ["train", "val"]

    train_percent = 0.9  # 训练验证集的占比

    """
    划分数据集,数据集的组织格式如下：
    --datasets_path
        --DUTS-TR
            --DUTS-TR-Image
                --1.jpg
                --2.jpg
                ...
            --DUTS-TR-Mask
                --1.png
                --2.png
                ...
        --DUTS-TE
            --DUTS-TE-Image
                --1.jpg
                --2.jpg
                ...
            --DUTS-TE-Mask
                --1.png
                --2.png
                ...
    """
    for dataset_name in data_category:
        print(dataset_name)
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if dataset_name == data_category[0]:
            print("训练集")
            img_name_list = os.listdir(os.path.join(dataset_path, data_category[0] + "-" + img_labels[0]))
            print("训练验证集数量 = ", len(img_name_list))

            np.random.shuffle(img_name_list)
            num_train_val = len(img_name_list)
            num_train = int(num_train_val * train_percent)
            num_val = num_train_val - num_train
            print("训练集图像数量 = %d (%.2f)" % (num_train, num_train / num_train_val))
            print("验证集图像数量 = %d (%.2f)" % (num_val, num_val / num_train_val))

            for dataset in datasets:
                if dataset == datasets[0]:
                    # 训练集
                    file_name_list = img_name_list[: num_train]
                elif dataset == datasets[1]:
                    # 验证集
                    file_name_list = img_name_list[num_train:]
                save_path = os.path.join(save_dir, "duts_seg_" + dataset + ".txt")

                list_file = open(save_path, 'w')
                for file_name in tqdm(file_name_list):
                    # 检查文件后缀
                    if os.path.splitext(file_name)[-1] != ".jpg":
                        continue
                    img_path = os.path.join(dataset_path, data_category[0] + "-" + img_labels[0], file_name)
                    mask_path = os.path.join(dataset_path, data_category[0] + "-" + img_labels[1],
                                             os.path.splitext(file_name)[0] + ".png")
                    list_file.write("%s" % img_path + ";" + '%s' % mask_path + ";")
                    list_file.write('\n')
                list_file.close()
        else:
            img_name_list = os.listdir(os.path.join(dataset_path, data_category[1] + "-" + img_labels[0]))
            num_test = len(img_name_list)
            print("测试集数量 = ", num_test)
            save_path = os.path.join(save_dir, "duts_seg_test.txt")
            list_file = open(save_path, 'w')
            for file_name in tqdm(img_name_list):
                # 检查文件后缀
                if os.path.splitext(file_name)[-1] != ".jpg":
                    continue
                img_path = os.path.join(dataset_path, data_category[1] + "-" + img_labels[0], file_name)
                mask_path = os.path.join(dataset_path, data_category[1] + "-" + img_labels[1],
                                         os.path.splitext(file_name)[0] + ".png")
                list_file.write("%s" % img_path + ";" + '%s' % mask_path + ";")
                list_file.write('\n')
            list_file.close()


def generate_coco_data_txt():
    """
    参数设置
    """
    # 注意train耗时长
    datasets = ["val"] #["val", "train"]train数据集数量 = 118287, 未标注数量 = 1021

    for dataset in datasets:
        coco_image_dir = fr"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\coco_2017\JPEGImages\{dataset}2017"
        coco_json_path = fr"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\coco_2017\Annotations\instances_{dataset}2017.json"
        txt_path = fr"D:\Code\Projects\python\AI_Model_Temp\data_txt\coco_det\coco_det_{dataset}.txt"

        # 读取json文件
        with open(coco_json_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)

        # 打开txt文件
        txt_data = open(txt_path, 'w', encoding='utf-8')

        # 检查图片规模
        json_image_num = len(json_data["images"])
        jpg_image_num = len(os.listdir(coco_image_dir))
        if json_image_num != jpg_image_num:
            print(f"{dataset}数据集 标签图片数量不等于实际图片数量: json_image_num = {json_image_num}， jpg_image_num = {jpg_image_num}")

        print(f"标签图片数量等于实际图片数量: {dataset}数据集数量 = {json_image_num}")
        num_no_label = 0
        # 遍历文件
        for image_data in tqdm(json_data["images"]):
            file_name = image_data["file_name"]
            image_path = os.path.join(coco_image_dir, file_name)
            if not os.path.exists(image_path):
                print("图片不存在: ", image_path)
                continue
            txt_data.write(f"{image_path};")
            # 获取图片id
            image_id = image_data["id"]
            # 遍历标签
            num_box = 0
            for annotation_data in json_data["annotations"]:
                if annotation_data["image_id"] == image_id:
                    # 类别
                    category_id = annotation_data["category_id"]
                    bbox = annotation_data["bbox"]
                    bbox_category_str = ",".join(str(box) for box in bbox) + "," + str(category_id) + ";"
                    txt_data.write(bbox_category_str)
                    num_box += 1
            if num_box == 0:
                num_no_label += 1
            txt_data.write('\n')
        print(f"{dataset}数据集数量 = {json_image_num}, 未标注数量 = {num_no_label}")
        txt_data.close()

def generate_voc_data_txt():
    """
    参数设置
    """
    train_percent = 0.8
    classes_txt_path = r"D:\Code\Projects\python\AI_Model_Temp\data_txt\voc_det\voc_classes.txt"
    class_name_list, len_class = get_classes(classes_txt_path)

    # 训练集
    train_val_image_dir = r"I:\Dataset\PASCAL_VOC2012\OpenDataLab___PASCAL_VOC2012\voc_2012\train_val\JPEGImages"
    train_val_xml_dir = r"I:\Dataset\PASCAL_VOC2012\OpenDataLab___PASCAL_VOC2012\voc_2012\train_val\Annotations"
    train_val_xml_path_list = glob.glob(os.path.join(train_val_xml_dir, "*.xml"))

    num_train_val = len(train_val_xml_path_list)
    np.random.shuffle(train_val_xml_path_list)  # 打乱
    num_train = int(num_train_val * train_percent)  # 训练集个数
    num_val = num_train_val - num_train  # 验证集个数
    print("训练、验证集数量 = ", num_train_val)
    print("训练集图像数量 = %d" % num_train)
    print("验证集图像数量 = %d" % num_val)

    # 训练集
    train_txt_path = r"D:\Code\Projects\python\AI_Model_Temp\data_txt\voc_det\voc_det_train.txt"
    voc_xml_to_txt(txt_path=train_txt_path,
                   xml_path_list=train_val_xml_path_list[:num_train],
                   image_dir=train_val_image_dir,
                   class_name_list=class_name_list)
    # 验证集
    val_txt_path = r"D:\Code\Projects\python\AI_Model_Temp\data_txt\voc_det\voc_det_val.txt"
    voc_xml_to_txt(txt_path=val_txt_path,
                   xml_path_list=train_val_xml_path_list[num_train:],
                   image_dir=train_val_image_dir,
                   class_name_list=class_name_list)

    # 测试集
    test_image_dir = r"I:\Dataset\PASCAL_VOC2012\OpenDataLab___PASCAL_VOC2012\voc_2012\test\JPEGImages"
    test_xml_dir = r"I:\Dataset\PASCAL_VOC2012\OpenDataLab___PASCAL_VOC2012\voc_2012\test\Annotations"
    test_xml_path_list = glob.glob(os.path.join(test_xml_dir, "*.xml"))
    num_test = len(test_xml_path_list)
    print("测试集图像数量 = %d" % num_test)
    test_txt_path = r"D:\Code\Projects\python\AI_Model_Temp\data_txt\voc_det\voc_det_test.txt"
    voc_xml_to_txt(txt_path=test_txt_path,
                   xml_path_list=test_xml_path_list,
                   image_dir=test_image_dir,
                   class_name_list=class_name_list)



def voc_xml_to_txt(txt_path, xml_path_list, image_dir, class_name_list):
    # 打开txt文件
    txt_data = open(txt_path, 'w', encoding='utf-8')
    for xml_path in tqdm(xml_path_list):
        xml_data = open(xml_path, "r", encoding='utf-8')
        xml_tree = xml.etree.ElementTree.parse(xml_data)
        xml_root = xml_tree.getroot()

        # 检查图片文件
        image_file_name = xml_root.find("filename").text
        image_path = os.path.join(image_dir, image_file_name)
        if not os.path.exists(image_path):
            continue
        # 路径
        txt_data.write(f"{image_path};")

        for det_label in xml_root.iter('object'):
            class_name = det_label.find('name').text
            if class_name not in class_name_list:
                continue
            # 类别
            category_id = class_name_list.index(class_name)

            # 边界框
            xml_bbox = det_label.find("bndbox")
            bbox_list = [float(xml_bbox.find('xmin').text),
                         float(xml_bbox.find('ymin').text),
                         float(xml_bbox.find('xmax').text),
                         float(xml_bbox.find('ymax').text)]
            bbox_category_str = ",".join(str(box) for box in bbox_list) + "," + str(category_id) + ";"
            txt_data.write(bbox_category_str)
        txt_data.write('\n')
    txt_data.close()







if __name__ == "__main__":
    print(__file__)
    # generate_sky_data_txt()
    # generate_animal_data_txt()
    # generate_cat_dog_data_txt()
    # generate_duts_data_txt()
    generate_coco_data_txt()
    # generate_voc_data_txt()
