import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional, cast

import requests
from flask import current_app

from core.entities.model_entities import ModelStatus, ModelWithProviderEntity, ProviderModelWithStatusEntity
from core.model_runtime.entities.model_entities import ModelType, ParameterRule
from core.model_runtime.model_providers import model_provider_factory
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.provider_manager import ProviderManager
from models.provider import ProviderType
from services.entities.model_provider_entities import (
    CustomConfigurationResponse,
    CustomConfigurationStatus,
    DefaultModelResponse,
    ModelWithProviderEntityResponse,
    ProviderResponse,
    ProviderWithModelsResponse,
    SimpleProviderEntityResponse,
    SystemConfigurationResponse,
)

logger = logging.getLogger(__name__)

# cdg:模型提供商服务
class ModelProviderService:
    """
    Model Provider Service
    """
    def __init__(self) -> None:
        # cdg:初始化模型提供商管理器
        self.provider_manager = ProviderManager()

    # cdg:获取提供商列表，具体实现思路：
    # 1. 获取当前工作区的所有提供商配置
    # 2. 遍历提供商配置，根据模型类型过滤提供商
    # 3. 创建提供商响应对象
    # 4. 返回提供商列表
    def get_provider_list(self, tenant_id: str, model_type: Optional[str] = None) -> list[ProviderResponse]:
        """
        get provider list.

        :param tenant_id: workspace id
        :param model_type: model type
        :return:
        """
        # cdg:根据租户ID获取当前工作区的所有提供商配置
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # cdg:遍历提供商配置，根据模型类型过滤提供商
        provider_responses = []
        for provider_configuration in provider_configurations.values():
            if model_type:
                model_type_entity = ModelType.value_of(model_type)
                if model_type_entity not in provider_configuration.provider.supported_model_types:
                    continue

            # cdg:创建提供商响应对象
            provider_response = ProviderResponse(
                provider=provider_configuration.provider.provider,
                label=provider_configuration.provider.label,
                description=provider_configuration.provider.description,
                icon_small=provider_configuration.provider.icon_small,
                icon_large=provider_configuration.provider.icon_large,
                background=provider_configuration.provider.background,
                help=provider_configuration.provider.help,
                supported_model_types=provider_configuration.provider.supported_model_types,
                configurate_methods=provider_configuration.provider.configurate_methods,
                provider_credential_schema=provider_configuration.provider.provider_credential_schema,
                model_credential_schema=provider_configuration.provider.model_credential_schema,
                preferred_provider_type=provider_configuration.preferred_provider_type,
                custom_configuration=CustomConfigurationResponse(
                    status=CustomConfigurationStatus.ACTIVE
                    if provider_configuration.is_custom_configuration_available()
                    else CustomConfigurationStatus.NO_CONFIGURE
                ),
                system_configuration=SystemConfigurationResponse(
                    enabled=provider_configuration.system_configuration.enabled,
                    current_quota_type=provider_configuration.system_configuration.current_quota_type,
                    quota_configurations=provider_configuration.system_configuration.quota_configurations,
                ),
            )

            provider_responses.append(provider_response)

        return provider_responses

    # cdg:根据租户ID和提供商名称获取提供商模型列表，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称过滤提供商模型列表
    # 3. 创建提供商模型响应对象
    # 4. 返回提供商模型列表
    def get_models_by_provider(self, tenant_id: str, provider: str) -> list[ModelWithProviderEntityResponse]:
        """
        get provider models.
        For the model provider page,
        only supports passing in a single provider to query the list of supported models.

        :param tenant_id:
        :param provider:
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider available models
        return [
            ModelWithProviderEntityResponse(model) for model in provider_configurations.get_models(provider=provider)
        ]

    # cdg:根据租户ID和提供商名称获取提供商凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 返回提供商凭证
    def get_provider_credentials(self, tenant_id: str, provider: str):
        """
        get provider credentials.
        """
        provider_configurations = self.provider_manager.get_configurations(tenant_id)
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        return provider_configuration.get_custom_credentials(obfuscated=True)

    # cdg:验证提供商凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 验证提供商凭证
    def provider_credentials_validate(self, tenant_id: str, provider: str, credentials: dict) -> None:
        """
        validate provider credentials.

        :param tenant_id:
        :param provider:
        :param credentials:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        provider_configuration.custom_credentials_validate(credentials)

    # cdg:保存提供商凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 保存提供商凭证
    def save_provider_credentials(self, tenant_id: str, provider: str, credentials: dict) -> None:
        """
        save custom provider config.

        :param tenant_id: workspace id
        :param provider: provider name
        :param credentials: provider credentials
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Add or update custom provider credentials.
        provider_configuration.add_or_update_custom_credentials(credentials)

    # cdg:删除提供商凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 删除提供商凭证
    def remove_provider_credentials(self, tenant_id: str, provider: str) -> None:
        """
        remove custom provider config.

        :param tenant_id: workspace id
        :param provider: provider name
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Remove custom provider credentials.
        provider_configuration.delete_custom_credentials()

    # cdg:根据租户ID、提供商名称、模型类型和模型名称获取模型凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 根据模型类型和模型名称获取模型凭证
    def get_model_credentials(self, tenant_id: str, provider: str, model_type: str, model: str):
        """
        get model credentials.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Get model custom credentials from ProviderModel if exists
        return provider_configuration.get_custom_model_credentials(
            model_type=ModelType.value_of(model_type), model=model, obfuscated=True
        )

    # cdg:验证模型凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 验证模型凭证
    def model_credentials_validate(
        self, tenant_id: str, provider: str, model_type: str, model: str, credentials: dict
    ) -> None:
        """
        validate model credentials.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :param credentials: model credentials
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Validate model credentials
        provider_configuration.custom_model_credentials_validate(
            model_type=ModelType.value_of(model_type), model=model, credentials=credentials
        )

    # cdg:保存模型凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 保存模型凭证
    def save_model_credentials(
        self, tenant_id: str, provider: str, model_type: str, model: str, credentials: dict
    ) -> None:
        """
        save model credentials.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :param credentials: model credentials
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Add or update custom model credentials
        provider_configuration.add_or_update_custom_model_credentials(
            model_type=ModelType.value_of(model_type), model=model, credentials=credentials
        )

    # cdg:删除模型凭证，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 删除模型凭证
    def remove_model_credentials(self, tenant_id: str, provider: str, model_type: str, model: str) -> None:
        """
        remove model credentials.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Remove custom model credentials
        provider_configuration.delete_custom_model_credentials(model_type=ModelType.value_of(model_type), model=model)

    # cdg:根据模型类型获取模型列表，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据模型类型过滤模型列表
    # 3. 创建提供商模型响应对象
    # 4. 返回提供商模型列表
    def get_models_by_model_type(self, tenant_id: str, model_type: str) -> list[ProviderWithModelsResponse]:
        """
        get models by model type.

        :param tenant_id: workspace id
        :param model_type: model type
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider available models
        models = provider_configurations.get_models(model_type=ModelType.value_of(model_type))

        # Group models by provider
        provider_models: dict[str, list[ModelWithProviderEntity]] = {}
        for model in models:
            if model.provider.provider not in provider_models:
                provider_models[model.provider.provider] = []

            if model.deprecated:
                continue

            if model.status != ModelStatus.ACTIVE:
                continue

            provider_models[model.provider.provider].append(model)

        # convert to ProviderWithModelsResponse list
        providers_with_models: list[ProviderWithModelsResponse] = []
        for provider, models in provider_models.items():
            if not models:
                continue

            first_model = models[0]

            providers_with_models.append(
                ProviderWithModelsResponse(
                    provider=provider,
                    label=first_model.provider.label,
                    icon_small=first_model.provider.icon_small,
                    icon_large=first_model.provider.icon_large,
                    status=CustomConfigurationStatus.ACTIVE,
                    models=[
                        ProviderModelWithStatusEntity(
                            model=model.model,
                            label=model.label,
                            model_type=model.model_type,
                            features=model.features,
                            fetch_from=model.fetch_from,
                            model_properties=model.model_properties,
                            status=model.status,
                            load_balancing_enabled=model.load_balancing_enabled,
                        )
                        for model in models
                    ],
                )
            )

        return providers_with_models

    # cdg:根据租户ID、提供商名称和模型名称获取模型参数规则，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 根据模型类型获取模型实例
    # 4. 获取模型参数规则
    def get_model_parameter_rules(self, tenant_id: str, provider: str, model: str) -> list[ParameterRule]:
        """
        get model parameter rules.
        Only supports LLM.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model: model name
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Get model instance of LLM
        model_type_instance = provider_configuration.get_model_type_instance(ModelType.LLM)
        model_type_instance = cast(LargeLanguageModel, model_type_instance)

        # fetch credentials
        credentials = provider_configuration.get_current_credentials(model_type=ModelType.LLM, model=model)

        if not credentials:
            return []

        # Call get_parameter_rules method of model instance to get model parameter rules
        return list(model_type_instance.get_parameter_rules(model=model, credentials=credentials))

    # cdg:根据租户ID和模型类型获取默认模型，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据模型类型获取默认模型
    # 3. 创建默认模型响应对象
    # 4. 返回默认模型
    def get_default_model_of_model_type(self, tenant_id: str, model_type: str) -> Optional[DefaultModelResponse]:
        """
        get default model of model type.

        :param tenant_id: workspace id
        :param model_type: model type
        :return:
        """
        model_type_enum = ModelType.value_of(model_type)
        result = self.provider_manager.get_default_model(tenant_id=tenant_id, model_type=model_type_enum)
        try:
            return (
                DefaultModelResponse(
                    model=result.model,
                    model_type=result.model_type,
                    provider=SimpleProviderEntityResponse(
                        provider=result.provider.provider,
                        label=result.provider.label,
                        icon_small=result.provider.icon_small,
                        icon_large=result.provider.icon_large,
                        supported_model_types=result.provider.supported_model_types,
                    ),
                )
                if result
                else None
            )
        except Exception as e:
            logger.info(f"get_default_model_of_model_type error: {e}")
            return None

    # cdg:更新默认模型，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据模型类型获取默认模型
    # 3. 更新默认模型
    def update_default_model_of_model_type(self, tenant_id: str, model_type: str, provider: str, model: str) -> None:
        """
        update default model of model type.

        :param tenant_id: workspace id
        :param model_type: model type
        :param provider: provider name
        :param model: model name
        :return:
        """
        model_type_enum = ModelType.value_of(model_type)
        self.provider_manager.update_default_model_record(
            tenant_id=tenant_id, model_type=model_type_enum, provider=provider, model=model
        )

    # cdg:获取模型提供商图标，具体实现思路：
    # 1. 根据提供商名称获取提供商实例
    # 2. 获取提供商图标
    # 3. 返回提供商图标
    def get_model_provider_icon(
        self, provider: str, icon_type: str, lang: str
    ) -> tuple[Optional[bytes], Optional[str]]:
        """
        get model provider icon.

        :param provider: provider name
        :param icon_type: icon type (icon_small or icon_large)
        :param lang: language (zh_Hans or en_US)
        :return:
        """
        provider_instance = model_provider_factory.get_provider_instance(provider)
        provider_schema = provider_instance.get_provider_schema()
        file_name: str | None = None

        if icon_type.lower() == "icon_small":
            if not provider_schema.icon_small:
                raise ValueError(f"Provider {provider} does not have small icon.")

            if lang.lower() == "zh_hans":
                file_name = provider_schema.icon_small.zh_Hans
            else:
                file_name = provider_schema.icon_small.en_US
        else:
            if not provider_schema.icon_large:
                raise ValueError(f"Provider {provider} does not have large icon.")

            if lang.lower() == "zh_hans":
                file_name = provider_schema.icon_large.zh_Hans
            else:
                file_name = provider_schema.icon_large.en_US
        if not file_name:
            return None, None

        root_path = current_app.root_path
        provider_instance_path = os.path.dirname(
            os.path.join(root_path, provider_instance.__class__.__module__.replace(".", "/"))
        )
        file_path = os.path.join(provider_instance_path, "_assets")
        file_path = os.path.join(file_path, file_name)

        if not os.path.exists(file_path):
            return None, None

        mimetype, _ = mimetypes.guess_type(file_path)
        mimetype = mimetype or "application/octet-stream"

        # read binary from file
        byte_data = Path(file_path).read_bytes()
        return byte_data, mimetype

    # cdg:切换首选提供商，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 切换首选提供商类型
    def switch_preferred_provider(self, tenant_id: str, provider: str, preferred_provider_type: str) -> None:
        """
        switch preferred provider.

        :param tenant_id: workspace id
        :param provider: provider name
        :param preferred_provider_type: preferred provider type
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Convert preferred_provider_type to ProviderType
        preferred_provider_type_enum = ProviderType.value_of(preferred_provider_type)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Switch preferred provider type
        provider_configuration.switch_preferred_provider_type(preferred_provider_type_enum)

    # cdg:启用模型，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 启用模型
    def enable_model(self, tenant_id: str, provider: str, model: str, model_type: str) -> None:
        """
        enable model.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model: model name
        :param model_type: model type
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Enable model
        provider_configuration.enable_model(model=model, model_type=ModelType.value_of(model_type))

    # cdg:禁用模型，具体实现思路：
    # 1. 根据租户ID获取当前工作区的所有提供商配置
    # 2. 根据提供商名称获取提供商配置
    # 3. 禁用模型
    def disable_model(self, tenant_id: str, provider: str, model: str, model_type: str) -> None:
        """
        disable model.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model: model name
        :param model_type: model type
        :return:
        """
        # Get all provider configurations of the current workspace
        provider_configurations = self.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f"Provider {provider} does not exist.")

        # Enable model
        provider_configuration.disable_model(model=model, model_type=ModelType.value_of(model_type))

    # cdg:提交模型免费额度申请，具体实现思路：
    # 1. 获取模型免费额度申请API密钥和基础URL
    # 2. 构建API请求URL
    # 3. 发送POST请求
    # 4. 处理响应结果
    def free_quota_submit(self, tenant_id: str, provider: str):
        api_key = os.environ.get("FREE_QUOTA_APPLY_API_KEY")
        api_base_url = os.environ.get("FREE_QUOTA_APPLY_BASE_URL", "")
        api_url = api_base_url + "/api/v1/providers/apply"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        response = requests.post(api_url, headers=headers, json={"workspace_id": tenant_id, "provider_name": provider})
        if not response.ok:
            logger.error(f"Request FREE QUOTA APPLY SERVER Error: {response.status_code} ")
            raise ValueError(f"Error: {response.status_code} ")

        if response.json()["code"] != "success":
            raise ValueError(f"error: {response.json()['message']}")

        rst = response.json()

        if rst["type"] == "redirect":
            return {"type": rst["type"], "redirect_url": rst["redirect_url"]}
        else:
            return {"type": rst["type"], "result": "success"}

    # cdg:验证模型免费额度资格，具体实现思路：
    # 1. 获取模型免费额度申请API密钥和基础URL
    # 2. 构建API请求URL
    # 3. 发送POST请求
    # 4. 处理响应结果
    def free_quota_qualification_verify(self, tenant_id: str, provider: str, token: Optional[str]):
        api_key = os.environ.get("FREE_QUOTA_APPLY_API_KEY")
        api_base_url = os.environ.get("FREE_QUOTA_APPLY_BASE_URL", "")
        api_url = api_base_url + "/api/v1/providers/qualification-verify"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        json_data = {"workspace_id": tenant_id, "provider_name": provider}
        if token:
            json_data["token"] = token
        response = requests.post(api_url, headers=headers, json=json_data)
        if not response.ok:
            logger.error(f"Request FREE QUOTA APPLY SERVER Error: {response.status_code} ")
            raise ValueError(f"Error: {response.status_code} ")

        rst = response.json()
        if rst["code"] != "success":
            raise ValueError(f"error: {rst['message']}")

        data = rst["data"]
        if data["qualified"] is True:
            return {"result": "success", "provider_name": provider, "flag": True}
        else:
            return {"result": "success", "provider_name": provider, "flag": False, "reason": data["reason"]}
