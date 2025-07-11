import logging
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .deploy import DeploymentConfig
from .enterprise import EnterpriseFeatureConfig
from .extra import ExtraServiceConfig
from .feature import FeatureConfig
from .middleware import MiddlewareConfig
from .packaging import PackagingInfo
from .remote_settings_sources import RemoteSettingsSource, RemoteSettingsSourceConfig, RemoteSettingsSourceName
from .remote_settings_sources.apollo import ApolloSettingsSource

logger = logging.getLogger(__name__)


class RemoteSettingsSourceFactory(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        current_state = self.current_state
        remote_source_name = current_state.get("REMOTE_SETTINGS_SOURCE_NAME")
        if not remote_source_name:
            return {}

        remote_source: RemoteSettingsSource | None = None
        match remote_source_name:
            case RemoteSettingsSourceName.APOLLO:
                remote_source = ApolloSettingsSource(current_state)
            case _:
                logger.warning(f"Unsupported remote source: {remote_source_name}")
                return {}

        d: dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = remote_source.get_field_value(field, field_name)
            field_value = remote_source.prepare_field_value(field_name, field, field_value, value_is_complex)
            if field_value is not None:
                d[field_key] = field_value

        return d

# cdg:定义一个配置 DifyConfig，用于集中管理DIFY的各种配置
class DifyConfig(
    # Packaging info
    PackagingInfo, # cdg: 打包配置信息
    # Deployment configs
    DeploymentConfig, # cdg: 部署配置信息
    # Feature configs
    FeatureConfig, # cdg: 功能配置信息
    # Middleware configs
    MiddlewareConfig, # cdg: 中间件配置信息
    # Extra service configs
    ExtraServiceConfig, # cdg: 额外服务配置信息
    # Remote source configs
    RemoteSettingsSourceConfig, # cdg: 远程设置源配置信息
    # Enterprise feature configs
    # **Before using, please contact business@dify.ai by email to inquire about licensing matters.**
    EnterpriseFeatureConfig, # cdg: 企业级功能配置信息
): 
    # cdg:注意，DIFY对现有配置进行了分类，后续有新的配置，请添加到对应的配置类中，便于后续的维护和扩展。


    # cdg:定义类属性model_config，其值为SettingsConfigDict的实例。这个配置字典用于指定如何读取配置文件（如 .env 文件），并设置一些属性：
    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )

    # Before adding any config,
    # please consider to arrange it in the proper config group of existed or added
    # for better readability and maintainability.
    # Thanks for your concentration and consideration.

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            RemoteSettingsSourceFactory(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
