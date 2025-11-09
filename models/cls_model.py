# -*- coding:utf-8 -*-
# @FileName  :cls_model.py
# @Time      :2025/10/6 18:19
# @Author    :yxl

"""
分类模型训练基本流程
"""

import os.path

import torch
import numpy as np
import cv2
from .base_model import BaseModel
from model_utils.registry import MODEL_REGISTRY
from tqdm import tqdm


@MODEL_REGISTRY.register()
class ClsModel(BaseModel):
    def __init__(self, opt):
        super(ClsModel, self).__init__(opt)

        self.model_opt = self.opt["model"]

        """
        ———————————————————————创建优化器———————————————————————
        """
        self.optimizer = torch.optim.SGD(self.net_g.parameters(),
                                         lr=0.002,
                                         momentum=0.9,
                                         dampening=0,
                                         weight_decay=3e-4)

        # self.optimizer = torch.optim.Adam(self.net_g.parameters(),
        #                                   lr=0.001,
        #                                   betas=(0.9, 0.999),
        #                                   eps=1e-08,
        #                                   weight_decay=1e-5,
        #                                   amsgrad=False)
        """
        ———————————————————————创建学习率调度器———————————————————————
        """
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                                mode='min',
        #                                                                factor=0.8,
        #                                                                patience=3,
        #                                                                threshold=0.0001,
        #                                                                threshold_mode='rel',
        #                                                                cooldown=0,
        #                                                                min_lr=0,
        #                                                                eps=1e-08)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                       T_max=self.opt["train"]["epoch"] // 2,
                                                                       eta_min=0.0002,
                                                                       last_epoch=-1)

        """
        ———————————————————————加载预训练权重———————————————————————
        """
        if self.model_opt["pretrain_weight"]:
            # 检查文件
            if os.path.exists(self.model_opt["pretrain_weight"]):
                pretrain_weight = torch.load(self.model_opt["pretrain_weight"], map_location=self.device,
                                             weights_only=True)
                self.net_g.load_state_dict(pretrain_weight['model_state_dict'])
            else:
                print("pretrain_weight file not exist")

    # 调度器是否需要指标
    def update_lr(self, val_loss):
        # 不需要val_loss
        self.lr_scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        return current_lr

    # 训练
    def train_one_step(self, batch_data):
        self.net_g.train()

        #  加载数据
        images, labels = batch_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        # 训练
        predictions = self.net_g(images)
        # 损失
        train_loss = self.loss(predictions, labels)
        # 指标
        self.metric.update(predictions, labels)

        train_loss.backward()
        self.optimizer.step()

        return train_loss

    # 验证
    def val_epoch(self, val_dataloader):
        self.metric.reset()  # 重置指标计算
        self.net_g.eval()  # 验证模式

        sum_val_loss = 0
        epoch_val_loss = 0
        epoch_val_metric = 0

        with torch.no_grad():
            progress_bar = tqdm(iterable=enumerate(val_dataloader, 1),
                                desc="val Epoch 1/1",
                                total=len(val_dataloader),
                                leave=True,
                                unit='step',
                                dynamic_ncols=True,
                                position=0)

            # start val
            for index_step, batch_data in progress_bar:
                #  加载数据
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 推理
                predictions = self.net_g(images)
                # 损失
                val_loss = self.loss(predictions, labels)
                # 指标
                self.metric.update(predictions, labels)

                sum_val_loss += val_loss.item()
                epoch_val_loss = sum_val_loss / index_step

                epoch_val_metric = self.get_epoch_metric()

                # 更新进度条信息
                loss_metric_info = f'val_loss: {epoch_val_loss:.4f} | val_metric: {epoch_val_metric:.4f}'
                progress_bar.set_postfix_str(f'{loss_metric_info}')
            # end val

            progress_bar.close()
            self.metric.reset()  # 重置指标计算

            return epoch_val_loss, epoch_val_metric

    # 测试
    def test_epoch(self, test_dataloader):
        self.metric.reset()  # 重置指标计算
        self.net_g.eval()  # 验证模式

        sum_test_loss = 0
        epoch_test_loss = 0
        epoch_test_metric = 0

        with torch.no_grad():
            progress_bar = tqdm(iterable=enumerate(test_dataloader, 1),
                                desc="test Epoch 1/1",
                                total=len(test_dataloader),
                                leave=True,
                                unit='step',
                                dynamic_ncols=True,
                                position=0)

            # start test
            for index_step, batch_data in progress_bar:
                #  加载数据
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 推理
                predictions = self.net_g(images)
                # 损失
                test_loss = self.loss(predictions, labels)
                # 指标
                self.metric.update(predictions, labels)

                sum_test_loss += test_loss.item()
                epoch_test_loss = sum_test_loss / index_step

                epoch_test_metric = self.get_epoch_metric()

                # 更新进度条信息
                loss_metric_info = f'test_loss: {epoch_test_loss:.4f} | test_metric: {epoch_test_metric:.4f}'
                progress_bar.set_postfix_str(f'{loss_metric_info}')
            # end test

            progress_bar.close()
            self.metric.reset()  # 重置指标计算

            return epoch_test_loss, epoch_test_metric

    # 保存可视化
    def save_visualization(self, epoch, data, visual_path):
        self.net_g.eval()
        # 推理
        with torch.no_grad():
            for index, in_tensor in enumerate(data[0]):
                in_tensor = in_tensor.to(self.device)
                out_tensor = self.net_g(in_tensor)
                out_tensor = torch.softmax(out_tensor, dim=-1)
                output_np = out_tensor.squeeze().cpu().numpy()
                # 标签
                pred_label = np.argmax(output_np)
                # 概率
                probability = np.max(output_np)

                save_path = os.path.join(visual_path, f"{index}_epoch_{epoch}_label_{pred_label}_{probability:.4f}.jpg")
                cv2.imwrite(save_path, data[1][index])


if __name__ == "__main__":
    print(__file__)