from configs import dify_config
from dify_app import DifyApp

# cdg: 检查是否启用压缩
def is_enabled() -> bool:
    return dify_config.API_COMPRESSION_ENABLED

# cdg: 初始化压缩
def init_app(app: DifyApp):
    from flask_compress import Compress  # type: ignore

    compress = Compress()
    compress.init_app(app)
