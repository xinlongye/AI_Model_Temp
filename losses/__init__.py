# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2025/10/6 15:47
# @Author    :yxl


import importlib
from copy import deepcopy
from os import path as osp

from model_utils.tools import scandir
from model_utils.registry import LOSS_REGISTRY

__all__ = ['build_loss']

# automatically scan and import loss modules for registry
# scan all the files under the 'data' folder and collect files ending with '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the loss modules
_loss_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]


def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    return loss

