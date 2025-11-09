# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2025/10/6 15:47
# @Author    :yxl


import importlib
from copy import deepcopy
from os import path as osp

from model_utils.tools import scandir
from model_utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import data_txt modules for registry
# scan all the files under the 'data' folder and collect files ending with '_dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(dataset_folder) if v.endswith('_dataset.py')]
# import all the data_txt modules
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]


def build_dataset(opt):
    opt = deepcopy(opt)
    dataset_type = opt.pop('type')
    dataset = DATASET_REGISTRY.get(dataset_type)(**opt)
    return dataset

