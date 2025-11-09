# -*- coding: utf-8 -*-
# @File    : tools.py
# @Time    : 2025/9/8 20:33
# @Author  : yxl
import os
import yaml
import torch
import torchvision
import platform
from tabulate import tabulate
import cv2
import numpy as np
import random


# 加载yaml文件
def yaml_load(yaml_path):
    # 检查路径
    if not os.path.exists(yaml_path):
        print("option yaml file not exists.")
        return None

    with open(yaml_path, "r") as f:
        option_data = yaml.load(f, Loader=yaml.FullLoader)
        return option_data


# 扫描目录
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path_in, suffix_in, recursive_in):
        for entry in os.scandir(dir_path_in):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix_in is None:
                    yield return_path
                elif return_path.endswith(suffix_in):
                    yield return_path
            else:
                if recursive_in:
                    yield from _scandir(entry.path, suffix_in=suffix_in, recursive_in=recursive_in)
                else:
                    continue

    return _scandir(dir_path, suffix_in=suffix, recursive_in=recursive)


# 获取torch、设备信息
def get_basic_info():
    # 初始化信息列表
    info = []

    # 框架信息
    info.extend([
        ["PyTorch", torch.__version__],
        ["TorchVision", torchvision.__version__],
        ["Python", platform.python_version()],
        ["system", f"{platform.system()} {platform.version()}"]
    ])

    # CUDA信息
    cuda_available = torch.cuda.is_available()
    info.append(["CUDA", cuda_available])

    if cuda_available:
        info.extend([
            ["CUDA", torch.version.cuda],
            ["cuDNN", torch.backends.cudnn.version()],
            ["GPU", torch.cuda.device_count()]
        ])

        # GPU信息
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # GB

            # 动态获取当前内存使用
            try:
                allocated_memory = torch.cuda.memory_allocated(i) / 1024 ** 3  # GB
                cached_memory = torch.cuda.memory_reserved(i) / 1024 ** 3  # GB
            except RuntimeError as e:
                allocated_memory = f"错误: {str(e)}"
                cached_memory = "N/A"

            info.extend([
                [f"GPU {i} info", ""],
                [f"  • name", gpu_name],
                [f"  • total memory    ", f"{total_memory:.2f} GB"],
                [f"  • allocated memory", f"{allocated_memory:.2f} GB"],
                [f"  • cache memory    ", f"{cached_memory:.2f} GB"]
            ])
            if i < torch.cuda.device_count() - 1:
                info.append(["-" * 30, "-" * 30])  # 分隔不同GPU
    else:
        info.append(["CUDA设备状态", "未检测到CUDA设备，使用CPU运行"])

    # CUDA模式信息
    info.extend([
        ["cuDNN 基准测试模式 ", torch.backends.cudnn.benchmark],
        ["cuDNN 确定性模式   ", torch.backends.cudnn.deterministic]
    ])

    # 打印表格 - 使用presto格式确保中文对齐
    return tabulate(info, headers=["属性", "值"], tablefmt="presto")


# 创建实验文件夹
def create_exp_folder(config_data, experiments_dir, experiment_category="train"):
    network_name = config_data["network"]["type"]
    dataset_name = config_data["dataset"]["train"]["type"]
    exp_name = config_data["experiment_name"]
    exp_file_name = "_".join([experiment_category, exp_name, network_name, dataset_name])
    # 实验目录
    experiment_path = os.path.join(experiments_dir, exp_file_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)

    if experiment_category == "train":
        # 检查点
        checkpoint_path = os.path.join(experiment_path, "checkpoint")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        # 可视化
        visualization_path = os.path.join(experiment_path, "visualization")
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path, exist_ok=True)

        # log
        logs_path = os.path.join(experiment_path, "logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)

        experiments_path_dict = {"checkpoint_path": checkpoint_path,
                                 "visualization_path": visualization_path,
                                 "logs_path": logs_path}
    else:
        # 可视化
        visualization_path = os.path.join(experiment_path, "visualization")
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path, exist_ok=True)

        # log
        logs_path = os.path.join(experiment_path, "logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)

        experiments_path_dict = {"visualization_path": visualization_path,
                                 "logs_path": logs_path}

    return experiments_path_dict


def get_gpu_memory_info():
    """获取显存使用情况信息"""
    if not torch.cuda.is_available():
        return "CPU Mode"

    device = torch.device("cuda")
    # 获取当前设备的显存信息
    device_id = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_id)

    # 计算显存使用情况（单位：GB）
    total_memory = properties.total_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)

    # 格式化为易读的字符串
    memory_info = f"GPU-{device_id}: allocated_{allocated_memory:.2f}GB_cached_{cached_memory:.2f}GB/{total_memory:.2f}GB"
    return memory_info


def get_classes(classes_path_in):
    with open(classes_path_in, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def letter_box(image, target_width, target_height):
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    letterboxed = cv2.copyMakeBorder(resized,
                                     top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=(128, 128, 128))

    return letterboxed

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print(__file__)
    get_basic_info()
