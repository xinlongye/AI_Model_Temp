# -*- coding:utf-8 -*-
# @FileName  :logger.py
# @Time      :2025/10/6 16:05
# @Author    :yxl
import logging

def init_logger(log_path, log_name, log_level=logging.INFO):
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 格式器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
