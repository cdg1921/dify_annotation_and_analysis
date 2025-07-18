from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

from configs import dify_config
from core.entities.model_entities import (
    ModelWithProviderEntity,
    ProviderModelWithStatusEntity,
)
from core.entities.provider_entities import QuotaConfiguration
from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.provider_entities import (
    ConfigurateMethod,
    ModelCredentialSchema,
    ProviderCredentialSchema,
    ProviderHelpEntity,
    SimpleProviderEntity,
)
from models.provider import ProviderQuotaType, ProviderType

# cdg: 定义CustomConfigurationStatus枚举类，用于表示自定义配置状态。
class CustomConfigurationStatus(Enum):
    """
    Enum class for custom configuration status.
    """

    ACTIVE = "active"
    NO_CONFIGURE = "no-configure"

# cdg: 定义CustomConfigurationResponse类，用于表示自定义配置响应。
class CustomConfigurationResponse(BaseModel):
    """
    Model class for provider custom configuration response.
    """

    status: CustomConfigurationStatus

# cdg: 定义SystemConfigurationResponse类，用于表示系统配置响应。
class SystemConfigurationResponse(BaseModel):
    """
    Model class for provider system configuration response.
    """

    enabled: bool
    current_quota_type: Optional[ProviderQuotaType] = None
    quota_configurations: list[QuotaConfiguration] = []

# cdg: 定义ProviderResponse类，用于表示提供商响应。
class ProviderResponse(BaseModel):
    """
    Model class for provider response.
    """

    provider: str
    label: I18nObject
    description: Optional[I18nObject] = None
    icon_small: Optional[I18nObject] = None
    icon_large: Optional[I18nObject] = None
    background: Optional[str] = None
    help: Optional[ProviderHelpEntity] = None
    supported_model_types: list[ModelType]
    configurate_methods: list[ConfigurateMethod]
    provider_credential_schema: Optional[ProviderCredentialSchema] = None
    model_credential_schema: Optional[ModelCredentialSchema] = None
    preferred_provider_type: ProviderType
    custom_configuration: CustomConfigurationResponse
    system_configuration: SystemConfigurationResponse

    # pydantic configs
    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **data) -> None:
        super().__init__(**data)

        url_prefix = dify_config.CONSOLE_API_URL + f"/console/api/workspaces/current/model-providers/{self.provider}"
        if self.icon_small is not None:
            self.icon_small = I18nObject(
                en_US=f"{url_prefix}/icon_small/en_US", zh_Hans=f"{url_prefix}/icon_small/zh_Hans"
            )

        if self.icon_large is not None:
            self.icon_large = I18nObject(
                en_US=f"{url_prefix}/icon_large/en_US", zh_Hans=f"{url_prefix}/icon_large/zh_Hans"
            )

# cdg: 定义ProviderWithModelsResponse类，用于表示提供商模型响应。
class ProviderWithModelsResponse(BaseModel):
    """
    Model class for provider with models response.
    """

    provider: str
    label: I18nObject
    icon_small: Optional[I18nObject] = None
    icon_large: Optional[I18nObject] = None
    status: CustomConfigurationStatus
    models: list[ProviderModelWithStatusEntity]

    def __init__(self, **data) -> None:
        super().__init__(**data)

        url_prefix = dify_config.CONSOLE_API_URL + f"/console/api/workspaces/current/model-providers/{self.provider}"
        if self.icon_small is not None:
            self.icon_small = I18nObject(
                en_US=f"{url_prefix}/icon_small/en_US", zh_Hans=f"{url_prefix}/icon_small/zh_Hans"
            )

        if self.icon_large is not None:
            self.icon_large = I18nObject(
                en_US=f"{url_prefix}/icon_large/en_US", zh_Hans=f"{url_prefix}/icon_large/zh_Hans"
            )

# cdg: 定义SimpleProviderEntityResponse类，用于表示简单提供商实体响应。
class SimpleProviderEntityResponse(SimpleProviderEntity):
    """
    Simple provider entity response.
    """

    def __init__(self, **data) -> None:
        super().__init__(**data)

        url_prefix = dify_config.CONSOLE_API_URL + f"/console/api/workspaces/current/model-providers/{self.provider}"
        if self.icon_small is not None:
            self.icon_small = I18nObject(
                en_US=f"{url_prefix}/icon_small/en_US", zh_Hans=f"{url_prefix}/icon_small/zh_Hans"
            )

        if self.icon_large is not None:
            self.icon_large = I18nObject(
                en_US=f"{url_prefix}/icon_large/en_US", zh_Hans=f"{url_prefix}/icon_large/zh_Hans"
            )

# cdg: 定义DefaultModelResponse类，用于表示默认模型响应。
class DefaultModelResponse(BaseModel):
    """
    Default model entity.
    """

    model: str
    model_type: ModelType
    provider: SimpleProviderEntityResponse

    # pydantic configs
    model_config = ConfigDict(protected_namespaces=())

# cdg: 定义ModelWithProviderEntityResponse类，用于表示模型提供商实体响应。
class ModelWithProviderEntityResponse(ModelWithProviderEntity):
    """
    Model with provider entity.
    """

    # FIXME type error ignore here
    provider: SimpleProviderEntityResponse  # type: ignore

    def __init__(self, model: ModelWithProviderEntity) -> None:
        super().__init__(**model.model_dump())
