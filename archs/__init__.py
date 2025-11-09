# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2025/10/6 15:47
# @Author    :yxl

import importlib
from copy import deepcopy
from os import path as osp

from model_utils.tools import scandir
from model_utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    return net
