# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2025/10/6 15:47
# @Author    :yxl


import importlib
from copy import deepcopy
from os import path as osp

from model_utils.tools import scandir
from model_utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'data' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt["model"]["type"])(opt)
    return model
