from pydantic import Field
from pydantic_settings import BaseSettings

# cdg: 打包构建信息
class PackagingInfo(BaseSettings):
    """
    Packaging build information
    """
    # cdg: 当前版本
    CURRENT_VERSION: str = Field(
        description="Dify version",
        default="0.15.1",
    )
    # cdg: 提交SHA-1
    COMMIT_SHA: str = Field(
        description="SHA-1 checksum of the git commit used to build the app",
        default="",
    )
