"""Abstract interface for file storage implementations."""

from abc import ABC, abstractmethod
from collections.abc import Generator

# cdg: 存储服务基类，定义了文件存储服务的基本接口，用于实现文件的存储、加载、下载、删除等操作。
class BaseStorage(ABC):
    """Interface for file storage."""

    # cdg: 保存文件
    @abstractmethod
    def save(self, filename, data):
        raise NotImplementedError

    # cdg: 一次性加载文件
    @abstractmethod
    def load_once(self, filename: str) -> bytes:
        raise NotImplementedError

    # cdg: 流式加载文件
    @abstractmethod
    def load_stream(self, filename: str) -> Generator:
        raise NotImplementedError

    # cdg: 下载文件
    @abstractmethod
    def download(self, filename, target_filepath):
        raise NotImplementedError

    # cdg: 检查文件是否存在
    @abstractmethod
    def exists(self, filename):
        raise NotImplementedError

    # cdg: 删除文件
    @abstractmethod
    def delete(self, filename):
        raise NotImplementedError
