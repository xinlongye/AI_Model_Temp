# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2025/10/6 15:47
# @Author    :yxl


import importlib
from copy import deepcopy
from os import path as osp

from model_utils.tools import scandir
from model_utils.registry import METRIC_REGISTRY

__all__ = ['build_metric']

# automatically scan and import metric modules for registry
# scan all the files under the 'data' folder and collect files ending with '_metric.py'
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_metric_modules = [importlib.import_module(f'metrics.{file_name}') for file_name in metric_filenames]


def build_metric(opt):
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**opt)
    return metric
