# -*- coding:utf-8 -*-
# @FileName  :train.py
# @Time      :2025/10/6 15:26
# @Author    :yxl

import torch
import shutil
from models import build_model
from data import build_dataset
import os
import yaml
import numpy as np
import random
from tqdm import tqdm
from model_utils.tools import get_gpu_memory_info, create_exp_folder, get_basic_info, seed_everything
from model_utils.logger import init_logger
import logging
import tempfile
tempfile.tempdir = "./temp"


def worker_init_fn(worker_id):
    seed = worker_id + 11
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def train_pipeline(opt_path, project_path):
    # 设置随机种子
    seed_everything(seed=11)

    # 读取配置文件
    with open(opt_path, "r", encoding="utf-8") as f:
        option_data = yaml.load(f, Loader=yaml.FullLoader)
    """
    ————————————————————————训练准备————————————————————————
    """
    start_epoch = 0
    train_lr = 0.001
    best_val_loss = 1000
    best_val_metric = 0
    result_log = {}

    train_option = option_data["train"]
    val_option = option_data["val"]
    dataset_option = option_data["dataset"]

    # 创建实验文件夹
    experiments_dir = os.path.join(project_path, "experiments")
    exp_path_dict = create_exp_folder(option_data, experiments_dir, "train")

    # 初始化log
    log_file_name = "_".join(["train", option_data["network"]["type"], dataset_option["train"]["type"]]) + ".log"
    log_path = os.path.join(exp_path_dict["logs_path"], log_file_name)
    logger = init_logger(log_path=log_path, log_name=option_data["experiment_name"],  log_level=logging.DEBUG)
    logger.info(get_basic_info())

    # 复制配置文件到实验文件夹
    copy_yaml_path = os.path.join(exp_path_dict["logs_path"], os.path.split(opt_path)[1])
    shutil.copy(opt_path, str(copy_yaml_path))
    """
    ————————————————————————加载数据————————————————————————
    """
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 4 if torch.cuda.is_available() else 1

    train_dataset = build_dataset(dataset_option["train"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_option["batch_size"],
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory,
                                                   drop_last=True,
                                                   worker_init_fn=worker_init_fn)

    val_dataset = build_dataset(dataset_option["val"])
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=val_option["batch_size"],
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory,
                                                 drop_last=True,
                                                 worker_init_fn=worker_init_fn)
    logger.info("----------------------start training----------------------")
    logger.debug(f"number of train dataset    = {len(train_dataset)}; steps of training dataset = {len(train_dataloader)}")
    logger.debug(f"number of val dataset      = {len(val_dataset)}; steps of val dataset      = {len(val_dataloader)}")


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
    if train_option["resume_train"]:
        # 读取给定的检查点文件, 若没有则读取最近文件
        if train_option["checkpoint_path"] == "":
            last_checkpoint_path = os.path.join(exp_path_dict["checkpoint_path"], str(option_data["model"]["type"]) + "_last_model.pt")
            if os.path.exists(last_checkpoint_path):
                start_epoch, best_val_loss, best_val_metric, train_lr = model.load_checkpoint(last_checkpoint_path)
                start_epoch += 1
                logger.info(f"load last checkpoint = {last_checkpoint_path}, start_epoch = {start_epoch}")
        else:
            if os.path.exists(train_option["checkpoint_path"]):
                start_epoch, best_val_loss, best_val_metric, train_lr = model.load_checkpoint(train_option["checkpoint_path"])
                start_epoch += 1
                logger.info("load config checkpoint = " + train_option["checkpoint_path"] + ", start_epoch = " + str(start_epoch))

    """
    ————————————————————————开始训练————————————————————————
    """
    for current_epoch in range(start_epoch, train_option["epoch"]):
        progress_bar = tqdm(iterable=enumerate(train_dataloader, 1),
                            desc=f'Epoch {current_epoch}/{train_option["epoch"]}',
                            total=len(train_dataloader),
                            leave=True,
                            unit='step',
                            dynamic_ncols=True,
                            position=0)

        sum_train_loss = 0
        epoch_train_loss = 0
        epoch_train_metric = 0

        # start epoch
        for index_step, batch_data in progress_bar:
            # 更新模型参数
            train_loss = model.train_one_step(batch_data)

            # 损失
            sum_train_loss += train_loss.detach().item()
            epoch_train_loss = sum_train_loss / index_step

            # 指标
            epoch_train_metric = model.get_epoch_metric()

            # 更新进度条信息
            gpu_info = get_gpu_memory_info()
            loss_metric_info = f'train_lr: {train_lr:.8f} | train_loss: {epoch_train_loss:.4f} | train_metric: {epoch_train_metric:.4f}'
            progress_bar.set_postfix_str(f'{gpu_info} | {loss_metric_info}')

        progress_bar.close()
        # end epoch

        """
        ————————————————————————开始验证————————————————————————
        """
        val_loss, val_metric = model.val_epoch(val_dataloader)

        """
        ————————————————————————可视化————————————————————————
        """
        if train_option["num_visual"] != 0 and current_epoch % train_option["visual_period"] == 0:
            # 获取可视化数据
            train_visual_path = os.path.join(exp_path_dict["visualization_path"], "train")
            if not os.path.exists(train_visual_path):
                os.makedirs(train_visual_path)
            train_visual_data = train_dataset.equivalent_preprocess(train_option["num_visual"], train_visual_path)
            model.save_visualization(current_epoch, train_visual_data, train_visual_path)


        if val_option["num_visual"] != 0 and current_epoch % val_option["visual_period"] == 0:
            # 获取可视化数据
            val_visual_path = os.path.join(exp_path_dict["visualization_path"], "val")
            if not os.path.exists(val_visual_path):
                os.makedirs(val_visual_path)
            val_visual_data = val_dataset.equivalent_preprocess(val_option["num_visual"], val_visual_path)
            model.save_visualization(current_epoch, val_visual_data, val_visual_path)


        """
        ————————————————————————更新学习率————————————————————————
        """
        current_lr = model.update_lr(val_loss)
        if train_lr != current_lr:
            logger.debug(f'update lr: {train_lr:.6f} → {current_lr:.6f}')
            train_lr = current_lr

        """
        ————————————————————————保存模型————————————————————————
        """
        # 获取训练状态
        state_dict = {
            'epoch': current_epoch,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'lr': current_lr}
        # 最佳损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_loss = True
            result_log["best_val_loss_epoch"] = current_epoch
            result_log["best_val_loss"] = best_val_loss
        else:
            save_best_loss = False

        # 最佳指标
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            save_best_metric = True
            result_log["best_val_metric_epoch"] = current_epoch
            result_log["best_val_metric"] = best_val_metric
        else:
            save_best_metric = False

        model.save_checkpoint(checkpoint_path=exp_path_dict["checkpoint_path"],
                              state_dict=state_dict,
                              save_best_loss=save_best_loss,
                              save_best_metric=save_best_metric)

        result_log["train_loss"] = epoch_train_loss
        result_log["train_metric"] = epoch_train_metric
        result_log["val_loss"] = val_loss
        result_log["val_metric"] = val_metric

        result_str = "|".join(f"{key}: {value}" if isinstance(value, int) else f"{key}: {value:.4f}" for key, value in result_log.items())
        logger.debug(result_str)

    # end train


if __name__ == "__main__":
    # gpu配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 配置路径
    config_path = r"D:\Code\Projects\python\AI_Model_Temp\options\seg_u2net_duts_option.yml"

    # 工程目录
    root_path = os.path.dirname(os.path.abspath(__file__))

    # 训练管道
    train_pipeline(opt_path=config_path, project_path=root_path)