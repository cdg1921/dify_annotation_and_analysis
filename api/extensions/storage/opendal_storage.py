import logging
import os
from collections.abc import Generator
from pathlib import Path

import opendal  # type: ignore[import]
from dotenv import dotenv_values

from extensions.storage.base_storage import BaseStorage

logger = logging.getLogger(__name__)

# cdg: 获取opendal的配置参数
def _get_opendal_kwargs(*, scheme: str, env_file_path: str = ".env", prefix: str = "OPENDAL_"):
    kwargs = {}
    config_prefix = prefix + scheme.upper() + "_"
    for key, value in os.environ.items():
        if key.startswith(config_prefix):
            kwargs[key[len(config_prefix) :].lower()] = value

    file_env_vars: dict = dotenv_values(env_file_path) or {}
    for key, value in file_env_vars.items():
        if key.startswith(config_prefix) and key[len(config_prefix) :].lower() not in kwargs and value:
            kwargs[key[len(config_prefix) :].lower()] = value

    return kwargs

# cdg: 实现本地文件存储服务，继承自BaseStorage类，实现了文件的存储、加载、下载、删除等操作。之所以改为opendal_storage
class OpenDALStorage(BaseStorage):
    def __init__(self, scheme: str, **kwargs):
        kwargs = kwargs or _get_opendal_kwargs(scheme=scheme)
        if scheme == "fs": # cdg: 如果scheme为fs，则创建本地文件存储服务
            root = kwargs.get("root", "storage") # cdg: 获取本地文件存储服务的根目录
            Path(root).mkdir(parents=True, exist_ok=True) # cdg: 创建本地文件存储服务的根目录

        self.op = opendal.Operator(scheme=scheme, **kwargs)
        logger.debug(f"opendal operator created with scheme {scheme}") # cdg: 创建opendal存储服务
        retry_layer = opendal.layers.RetryLayer(max_times=3, factor=2.0, jitter=True) # cdg: 创建重试层
        self.op = self.op.layer(retry_layer) # cdg: 添加重试层
        logger.debug("added retry layer to opendal operator")

    # cdg: 保存文件
    def save(self, filename: str, data: bytes) -> None:
        self.op.write(path=filename, bs=data)
        logger.debug(f"file {filename} saved")

    # cdg: 一次性加载文件
    def load_once(self, filename: str) -> bytes:
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        content: bytes = self.op.read(path=filename)
        logger.debug(f"file {filename} loaded")
        return content

    # cdg: 流式加载文件
    def load_stream(self, filename: str) -> Generator:
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        batch_size = 4096
        file = self.op.open(path=filename, mode="rb")
        while chunk := file.read(batch_size):
            yield chunk
        logger.debug(f"file {filename} loaded as stream")

    # cdg: 下载文件
    def download(self, filename: str, target_filepath: str):
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        with Path(target_filepath).open("wb") as f:
            f.write(self.op.read(path=filename))
        logger.debug(f"file {filename} downloaded to {target_filepath}")

    # cdg: 检查文件是否存在
    def exists(self, filename: str) -> bool:
        # FIXME this is a workaround for opendal python-binding do not have a exists method and no better
        # error handler here when opendal python-binding has a exists method, we should use it
        # more https://github.com/apache/opendal/blob/main/bindings/python/src/operator.rs
        try:
            res: bool = self.op.stat(path=filename).mode.is_file()
            logger.debug(f"file {filename} checked")
            return res
        except Exception:
            return False

    # cdg: 删除文件
    def delete(self, filename: str):
        if self.exists(filename):
            self.op.delete(path=filename)
            logger.debug(f"file {filename} deleted")
            return
        logger.debug(f"file {filename} not found, skip delete")
