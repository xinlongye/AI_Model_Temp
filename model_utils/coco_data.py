# -*- coding:utf-8 -*-
# @FileName  :coco_data.py
# @Time      :2025/10/6 20:20
# @Author    :yxl
import cv2
import json
import os

def run():
    img = cv2.imread(r"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\val2017\000000000139.jpg")
    print(img.shape)

def parse_coco_json():
    val_image_dir = r"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\coco_2017\JPEGImages\val2017"
    val_json_path = r"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\coco_2017\Annotations\instances_val2017.json"
    with open(val_json_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    print(json_data.keys())
    print(len(json_data["images"]))

    print(len(json_data["annotations"]))

    # 第一张图像
    print("第一张图像")
    image_idx = 888
    print(json_data["images"][image_idx].keys())
    print("json_data[images][image_idx] = ", json_data["images"][image_idx])
    image_path = os.path.join(val_image_dir, json_data["images"][image_idx]["file_name"])
    print(image_path)

    image = cv2.imread(image_path)
    print(image.shape)

    num_label = 1
    for i in json_data["annotations"]:
        if i["image_id"] == json_data["images"][image_idx]["id"]:
            print(i)
            print(json_data["images"][image_idx])
            print(f"有{num_label}标签")
            num_label += 1
            print(i["category_id"])

            x1y1 = (int(i["bbox"][0]), int(i["bbox"][1]))
            x2y2 = (int(i["bbox"][0] + i["bbox"][2]), int(i["bbox"][1] + i["bbox"][3]))
            image = cv2.rectangle(image, x1y1, x2y2, (0, 0, 255), 2)

    cv2.imwrite("1.jpg", image)

    # 第一个标签
    print("第一个标签")
    print(json_data["annotations"][0].keys())
    print(json_data["annotations"][0])


def parse_coco_test_json():
    test_json_path = r"I:\Dataset\COCO_2017\OpenDataLab___COCO_2017\coco_2017\Annotations\instances_test2017.json"
    with open(test_json_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    print(json_data.keys())

def parse_voc():
    print("解析voc数据集")


if __name__ == "__main__":
    print(__file__)
    parse_voc()
