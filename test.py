# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2025/10/6 15:26
# @Author    :yxl

import torch
import shutil
from models import build_model
from data import build_dataset
import os
import yaml
from model_utils.tools import create_exp_folder
from model_utils.logger import init_logger
import logging
import tempfile
tempfile.tempdir = "./temp"

def test_pipeline(opt_path, project_path):
    # 读取配置文件
    with open(opt_path, "r", encoding="utf-8") as f:
        option_data = yaml.load(f, Loader=yaml.FullLoader)

    test_option = option_data["test"]
    dataset_option = option_data["dataset"]

    # 创建实验文件夹
    experiments_dir = os.path.join(project_path, "experiments")
    exp_path_dict = create_exp_folder(option_data, experiments_dir, "test")

    # 初始化log
    log_file_name = "_".join(["test", option_data["network"]["type"], option_data["dataset"]["test"]["type"]]) + ".log"
    log_path = os.path.join(exp_path_dict["logs_path"], log_file_name)
    logger = init_logger(log_path=log_path, log_name=option_data["experiment_name"],  log_level=logging.DEBUG)

    # 复制配置文件到实验文件夹
    copy_yaml_path = os.path.join(exp_path_dict["logs_path"], os.path.split(opt_path)[1])
    shutil.copy(opt_path, str(copy_yaml_path))

    """
    ————————————————————————加载数据————————————————————————
    """
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 1 if torch.cuda.is_available() else 0

    test_dataset = build_dataset(dataset_option["test"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=test_option["batch_size"],
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory,
                                                 drop_last=True)
    logger.info("----------------------start testing----------------------")
    logger.debug(f"number of test dataset    = {len(test_dataset)}; steps of testing dataset = {len(test_dataloader)}")

    """
    ————————————————————————创建模型————————————————————————
    """
    model = build_model(option_data)

    """
    ————————————————————————恢复检查点————————————————————————
    约定：
    state_dict = {"model_state_dict": 模型权重,
                  "optimizer_state_dict": 优化器,
                  "scheduler_state_dict": 调度器,
                  "epoch": 当前epoch的索引,
                  "val_loss": 验证损失,
                  "val_metric": 验证指标
                  "lr": 学习率
    }
    最近的检查点：last_checkpoint.pt
    最好的检查点：best_checkpoint.pt
    """


    if test_option["checkpoint_path"] == "":
        # 读取指标最好的检查点
        last_checkpoint_path = os.path.join(exp_path_dict["checkpoint_path"], str(option_data["model"]["type"]) + "_best_metric_model.pt")
        if os.path.exists(last_checkpoint_path):
            model.load_checkpoint(last_checkpoint_path)
            logger.info(f"load last checkpoint = {last_checkpoint_path}")
    else:
        # 读取给定的检查点文件
        if os.path.exists(test_option["checkpoint_path"]):
            model.load_checkpoint(test_option["checkpoint_path"])
            logger.info("load config checkpoint = " + test_option["checkpoint_path"])

    """
    ————————————————————————开始测试————————————————————————
    """
    test_loss, test_metric = model.test_epoch(test_dataloader)
    result_str = f"test_loss: {test_loss:.4f} | test_metric: {test_metric:.4f}"
    logger.debug(result_str)

    """
    ————————————————————————可视化————————————————————————
    """
    if test_option["num_visual"] != 0:
        # 获取可视化数据
        test_visual_path = os.path.join(exp_path_dict["visualization_path"], "test")
        test_visual_data = test_dataset.equivalent_preprocess(test_option["num_visual"], test_visual_path)
        if not os.path.exists(test_visual_path):
            os.makedirs(test_visual_path)
        model.save_visualization(0, test_visual_data, test_visual_path)

    # end test


if __name__ == "__main__":
    # gpu配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 配置路径
    # config_path = r"/chengdu/yexinlong/AI_Model_Temp/options/seg_u2net_duts_option.yml"
    config_path = r"/chengdu/yexinlong/AI_Model_Temp/options/cls_sample_option.yml"

    # 工程目录
    root_path = os.path.dirname(os.path.abspath(__file__))

    # 训练管道
    test_pipeline(opt_path=config_path, project_path=root_path)
