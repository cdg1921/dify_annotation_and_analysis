from typing import Optional

from pydantic import Field

from .apollo import ApolloSettingsSourceInfo
from .base import RemoteSettingsSource
from .enums import RemoteSettingsSourceName

# cdg： 在项目通过引入Apollo相关的类和配置，实现了可以从Apollo配置中心动态获取和管理配置信息。这对于微服务架构、需要灵活配置和热更新的场景非常有用。
# cdg: 远程配置源配置信息,继承ApolloSettingsSourceInfo。
class RemoteSettingsSourceConfig(ApolloSettingsSourceInfo):
    REMOTE_SETTINGS_SOURCE_NAME: RemoteSettingsSourceName | str = Field(
        description="name of remote config source",
        default="",
    )


__all__ = ["RemoteSettingsSource", "RemoteSettingsSourceConfig", "RemoteSettingsSourceName"]
