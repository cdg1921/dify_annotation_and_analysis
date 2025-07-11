from collections.abc import Mapping
from typing import Any, Optional

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from configs.remote_settings_sources.base import RemoteSettingsSource

from .client import ApolloClient

# cdg: Apollo指的是一个远程配置中心项目，常用于集中式管理应用配置。它并不是指“阿波罗登月计划”或其他通用含义，而是特指由携程开源的Apollo配置中心。
# cdg: Apollo 配置中心是一个分布式配置管理平台，主要功能包括：
# cdg: 集中管理应用配置：所有服务的配置都可以在 Apollo 平台上统一管理。
# cdg: 动态推送配置变更：配置变更后可以实时推送到各个服务，无需重启应用。
# cdg: 多环境、多集群支持：可以为不同环境（如开发、测试、生产）和不同集群管理不同的配置。
# cdg: 权限与审计：支持配置的权限管理和变更历史审计。


# cdg: Apollo配置信息
class ApolloSettingsSourceInfo(BaseSettings):
    """
    Packaging build information
    """

    APOLLO_APP_ID: Optional[str] = Field(
        description="apollo app_id",
        default=None,
    )

    APOLLO_CLUSTER: Optional[str] = Field(
        description="apollo cluster",
        default=None,
    )

    APOLLO_CONFIG_URL: Optional[str] = Field(
        description="apollo config url",
        default=None,
    )

    APOLLO_NAMESPACE: Optional[str] = Field(
        description="apollo namespace",
        default=None,
    )


class ApolloSettingsSource(RemoteSettingsSource):
    def __init__(self, configs: Mapping[str, Any]):
        self.client = ApolloClient(
            app_id=configs["APOLLO_APP_ID"],
            cluster=configs["APOLLO_CLUSTER"],
            config_url=configs["APOLLO_CONFIG_URL"],
            start_hot_update=False,
            _notification_map={configs["APOLLO_NAMESPACE"]: -1},
        )
        self.namespace = configs["APOLLO_NAMESPACE"]
        self.remote_configs = self.client.get_all_dicts(self.namespace)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        if not isinstance(self.remote_configs, dict):
            raise ValueError(f"remote configs is not dict, but {type(self.remote_configs)}")
        field_value = self.remote_configs.get(field_name)
        return field_value, field_name, False
