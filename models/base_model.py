# -*- coding:utf-8 -*-
# @FileName  :base_model.py
# @Time      :2025/10/6 16:12
# @Author    :yxl


import torch
from archs import build_network
from losses import build_loss
from metrics import build_metric
import os


class BaseModel:
    def __init__(self, opt):
        """
        ——————————————————————设置一般参数———————————————————————
        """
        # 配置文件
        self.lr_scheduler = None
        self.optimizer = None
        self.opt = opt
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        ———————————————————————创建网络———————————————————————
        """
        self.net_g = build_network(opt['network'])
        self.net_g = self.net_g.to(self.device)

        """
        ———————————————————————创建损失函数———————————————————————
        """
        self.loss = build_loss(opt["loss"])

        """
        ———————————————————————创建指标计算———————————————————————
        """
        self.metric = build_metric(opt["metric"])
        self.metric.to(self.device)

    # 加载检查点
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.net_g.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['val_metric'], checkpoint['lr']

    # 获取平均指标
    def get_epoch_metric(self):
        return self.metric.compute()

    # 调度器是否需要指标
    def update_lr(self, val_loss):
        pass

    # 训练一个step
    def train_one_step(self, batch_data):
        pass

    # 验证
    def val_epoch(self, val_dataloader):
        pass

    # 测试
    def test_epoch(self, test_dataloader):
        pass

    # 保存训练状态
    def save_checkpoint(self, checkpoint_path, state_dict, save_best_loss, save_best_metric):
        state_dict['model_state_dict'] = self.net_g.state_dict()
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if save_best_loss:
            best_loss_path = os.path.join(checkpoint_path, str(self.opt["model"]["type"]) + "_best_loss_model.pt")
            torch.save(state_dict, best_loss_path)

        if save_best_metric:
            best_metric_path = os.path.join(checkpoint_path, str(self.opt["model"]["type"]) + "_best_metric_model.pt")
            torch.save(state_dict, best_metric_path)

        last_model_path = os.path.join(checkpoint_path, str(self.opt["model"]["type"]) + "_last_model.pt")
        torch.save(state_dict, last_model_path)

        # 带损失、指标的不覆盖的模型, 间隔保存
        current_epoch = state_dict["epoch"]
        if current_epoch % self.opt["model"]["save_period"] == 0:
            model_name = self.opt["model"]["type"]
            val_loss = state_dict["val_loss"]
            val_metric = state_dict["val_metric"]
            save_model_name = f'{model_name}_epoch_{current_epoch}_val_loss_{val_loss:.4f}_val_metric_{val_metric:.4f}.pt'
            save_model_path = os.path.join(checkpoint_path, save_model_name)
            torch.save(state_dict, save_model_path)
            print("not over write model save =  ", save_model_path)

    # 保存可视化
    def save_visualization(self, epoch, data, visual_path):
        pass


if __name__ == "__main__":
    print(__file__)
