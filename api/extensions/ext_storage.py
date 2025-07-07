import logging
from collections.abc import Callable, Generator
from typing import Literal, Union, overload

from flask import Flask

from configs import dify_config
from dify_app import DifyApp
from extensions.storage.base_storage import BaseStorage
from extensions.storage.storage_type import StorageType

logger = logging.getLogger(__name__)

# cdg: 存储扩展，用于管理存储，从而实现存储的持久化。
class Storage:
    def init_app(self, app: Flask):
        # cdg: 获取存储工厂，用于创建存储实例。
        storage_factory = self.get_storage_factory(dify_config.STORAGE_TYPE)
        with app.app_context(): # cdg: 在Flask应用的上下文中创建存储实例。
            self.storage_runner = storage_factory() # cdg: 创建存储实例。

    # cdg: 获取存储工厂，用于创建存储实例。
    @staticmethod
    def get_storage_factory(storage_type: str) -> Callable[[], BaseStorage]: # cdg: 获取存储工厂，用于创建存储实例。
        match storage_type: # cdg: 根据存储类型创建存储实例。
            case StorageType.S3:
                from extensions.storage.aws_s3_storage import AwsS3Storage

                return AwsS3Storage
            case StorageType.OPENDAL:
                from extensions.storage.opendal_storage import OpenDALStorage

                return lambda: OpenDALStorage(dify_config.OPENDAL_SCHEME)
            case StorageType.LOCAL: # cdg: 本地存储，用于存储本地文件。以前版本中为local_storage,新版本中为opendal_storage。
                from extensions.storage.opendal_storage import OpenDALStorage

                return lambda: OpenDALStorage(scheme="fs", root=dify_config.STORAGE_LOCAL_PATH)
            case StorageType.AZURE_BLOB:
                from extensions.storage.azure_blob_storage import AzureBlobStorage

                return AzureBlobStorage
            case StorageType.ALIYUN_OSS:
                from extensions.storage.aliyun_oss_storage import AliyunOssStorage

                return AliyunOssStorage
            case StorageType.GOOGLE_STORAGE:
                from extensions.storage.google_cloud_storage import GoogleCloudStorage

                return GoogleCloudStorage
            case StorageType.TENCENT_COS:
                from extensions.storage.tencent_cos_storage import TencentCosStorage

                return TencentCosStorage
            case StorageType.OCI_STORAGE:
                from extensions.storage.oracle_oci_storage import OracleOCIStorage

                return OracleOCIStorage
            case StorageType.HUAWEI_OBS:
                from extensions.storage.huawei_obs_storage import HuaweiObsStorage

                return HuaweiObsStorage
            case StorageType.BAIDU_OBS:
                from extensions.storage.baidu_obs_storage import BaiduObsStorage

                return BaiduObsStorage
            case StorageType.VOLCENGINE_TOS:
                from extensions.storage.volcengine_tos_storage import VolcengineTosStorage

                return VolcengineTosStorage
            case StorageType.SUPBASE:
                from extensions.storage.supabase_storage import SupabaseStorage

                return SupabaseStorage
            case _:
                raise ValueError(f"unsupported storage type {storage_type}")
    
    # cdg: 保存文件。
    def save(self, filename, data):
        try:
            self.storage_runner.save(filename, data)
        except Exception as e:
            logger.exception(f"Failed to save file {filename}")
            raise e

    @overload
    def load(self, filename: str, /, *, stream: Literal[False] = False) -> bytes: ...

    @overload
    def load(self, filename: str, /, *, stream: Literal[True]) -> Generator: ...

    # cdg: 加载文件，这里的Load函数重载了两次，第一次是根据stream参数的不同，第二次是根据filename参数的不同。
    def load(self, filename: str, /, *, stream: bool = False) -> Union[bytes, Generator]:
        try:
            if stream:
                return self.load_stream(filename)
            else:
                return self.load_once(filename)
        except Exception as e:
            logger.exception(f"Failed to load file {filename}")
            raise e

    # cdg: 一次性加载文件
    def load_once(self, filename: str) -> bytes:
        try:
            return self.storage_runner.load_once(filename)
        except Exception as e:
            logger.exception(f"Failed to load_once file {filename}")
            raise e
    
    # cdg: 流式加载文件
    def load_stream(self, filename: str) -> Generator:
        try:
            return self.storage_runner.load_stream(filename)
        except Exception as e:
            logger.exception(f"Failed to load_stream file {filename}")
            raise e
    # cdg: 下载文件
    def download(self, filename, target_filepath):
        try:
            self.storage_runner.download(filename, target_filepath)
        except Exception as e:
            logger.exception(f"Failed to download file {filename}")
            raise e

    # cdg: 检查文件是否存在
    def exists(self, filename):
        try:
            return self.storage_runner.exists(filename)
        except Exception as e:
            logger.exception(f"Failed to check file exists {filename}")
            raise e

    # cdg: 删除文件
    def delete(self, filename):
        try:
            return self.storage_runner.delete(filename)
        except Exception as e:
            logger.exception(f"Failed to delete file {filename}")
            raise e

# cdg: 创建存储实例
storage = Storage()

# cdg: 这个函数的作用是初始化存储应用，便于在Flask应用中使用存储
def init_app(app: DifyApp):
    storage.init_app(app)
