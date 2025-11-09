# -*- coding:utf-8 -*-
# @FileName  :basic_metric.py
# @Time      :2025/10/6 16:27
# @Author    :yxl


from torchmetrics import segmentation
from model_utils.registry import METRIC_REGISTRY
import torchmetrics


@METRIC_REGISTRY.register()
def BinaryMIoUMetric():
    # 需要将预测的mask转换为二值图（0, 1）
    # 预测值与目标值都为整数张量
    miou = torchmetrics.segmentation.MeanIoU(num_classes=2,
                                             include_background=False,
                                             per_class=True,
                                             input_format="index")
    return miou


@METRIC_REGISTRY.register()
def MultiClsAccuracy(num_classes=90, top_k=1):
    # 预测值可以为浮点型也可以为整型
    # 目标值为整型
    multi_cls_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, top_k=top_k)
    return multi_cls_acc
