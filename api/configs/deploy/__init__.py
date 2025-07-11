from pydantic import Field
from pydantic_settings import BaseSettings

# cdg: 部署配置信息
class DeploymentConfig(BaseSettings):
    """
    Configuration settings for application deployment
    """
    # cdg: 应用名称
    APPLICATION_NAME: str = Field(
        description="Name of the application, used for identification and logging purposes",
        default="langgenius/dify",
    )

    # cdg: 调试模式
    DEBUG: bool = Field(
        description="Enable debug mode for additional logging and development features",
        default=False,
    )

    # cdg: 部署版本
    EDITION: str = Field(
        description="Deployment edition of the application (e.g., 'SELF_HOSTED', 'CLOUD')",
        default="SELF_HOSTED",
    )

    # cdg: 部署环境
    DEPLOY_ENV: str = Field(
        description="Deployment environment (e.g., 'PRODUCTION', 'DEVELOPMENT'), default to PRODUCTION",
        default="PRODUCTION",
    )
