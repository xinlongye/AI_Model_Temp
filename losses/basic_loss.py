# -*- coding:utf-8 -*-
# @FileName  :basic_loss.py
# @Time      :2025/10/6 16:29
# @Author    :yxl


import torch
from torch.nn import functional as F
from model_utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']

@LOSS_REGISTRY.register()
class L1Loss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * F.l1_loss(input=pred, target=target, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * F.mse_loss(input=pred, target=target, reduction=self.reduction)


@LOSS_REGISTRY.register()
class BCELoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(BCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.bce_loss = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss_weight * self.bce_loss(pred, target)

@LOSS_REGISTRY.register()
class CELogitsLoss(torch.nn.Module):
    # 要求模型输出未经过softmax处理
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CELogitsLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss_weight * self.ce_loss(pred, target)

@LOSS_REGISTRY.register()
class CESoftmaxLoss(torch.nn.Module):
    # 要求模型输出经过softmax, loss中内置了log处理
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CESoftmaxLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.nll_loss = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, pred, target):
        pred_log = torch.log(pred)
        return self.loss_weight * self.nll_loss(pred_log, target)

@LOSS_REGISTRY.register()
class U2NetLoss(torch.nn.Module):
    def __init__(self, loss_weight=None):
        super(U2NetLoss, self).__init__()
        if loss_weight is None:
            loss_weight = [1.0, 0.7, 0.6, 0.5, 0.3, 0.2, 0.3]
        self.bce_loss = torch.nn.BCELoss(reduction="mean")
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss0 = self.loss_weight[0] * self.bce_loss(pred[0], target)
        loss1 = self.loss_weight[1] * self.bce_loss(pred[1], target)
        loss2 = self.loss_weight[2] * self.bce_loss(pred[2], target)
        loss3 = self.loss_weight[3] * self.bce_loss(pred[3], target)
        loss4 = self.loss_weight[4] * self.bce_loss(pred[4], target)
        loss5 = self.loss_weight[5] * self.bce_loss(pred[5], target)
        loss6 = self.loss_weight[6] * self.bce_loss(pred[6], target)

        depth_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return depth_loss
