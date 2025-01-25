import logging
import os
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from core.helper.module_import_helper import load_single_subclass_from_source
from core.helper.position_helper import get_provider_position_map, sort_to_dict_by_position_map
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.provider_entities import ProviderConfig, ProviderEntity, SimpleProviderEntity
from core.model_runtime.model_providers.__base.model_provider import ModelProvider
from core.model_runtime.schema_validators.model_credential_schema_validator import ModelCredentialSchemaValidator
from core.model_runtime.schema_validators.provider_credential_schema_validator import ProviderCredentialSchemaValidator

logger = logging.getLogger(__name__)


class ModelProviderExtension(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_instance: ModelProvider
    name: str
    position: Optional[int] = None


class ModelProviderFactory:
    # cdg:{provider_name:ModelProviderExtension}
    model_provider_extensions: Optional[dict[str, ModelProviderExtension]] = None

    def __init__(self) -> None:
        # for cache in memory
        self.get_providers()

    def get_providers(self) -> Sequence[ProviderEntity]:
        """
        Get all providers
        :return: list of providers
        """
        # scan all providers
        # cdg:{provider_name:model_provider_extension}
        model_provider_extensions = self._get_model_provider_map()

        # traverse all model_provider_extensions
        providers = []
        # cdg:对于每一个供应商model_provider_extension
        for model_provider_extension in model_provider_extensions.values():
            # cdg:从model_provider_extension中获取model_provider_instance
            # get model_provider instance
            model_provider_instance = model_provider_extension.provider_instance

            # cdg:获取供应商配置信息provider_schema，从provider_schema中获取供应商支持的模型类别supported_model_types，如llm、Embedding
            # get provider schema
            provider_schema = model_provider_instance.get_provider_schema()

            for model_type in provider_schema.supported_model_types:
                # get predefined models for given model type
                # cdg:遍历供应商目录下所有获取预置模型
                models = model_provider_instance.models(model_type)
                if models:
                    # cdg:将预置模型填充到provider_schema中，实例化了供应商模型信息
                    provider_schema.models.extend(models)

            providers.append(provider_schema)

        # cdg:providers是所有供应商provider_schema列表，provider_schema来源于每个供应商目录下的yaml文件，并补充了预定于的AI模型信息
        # return providers
        return providers

    def provider_credentials_validate(self, *, provider: str, credentials: dict) -> dict:
        """
        Validate provider credentials

        :param provider: provider name
        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        :return:
        """
        # get the provider instance
        model_provider_instance = self.get_provider_instance(provider)

        # get provider schema
        provider_schema = model_provider_instance.get_provider_schema()

        # get provider_credential_schema and validate credentials according to the rules
        provider_credential_schema = provider_schema.provider_credential_schema

        if not provider_credential_schema:
            raise ValueError(f"Provider {provider} does not have provider_credential_schema")

        # validate provider credential schema
        # cdg:获取yaml文件中预定于的provider_credential_schema
        validator = ProviderCredentialSchemaValidator(provider_credential_schema)
        # cdg:将用户提供的credentials与预置的provider_credential_schema对比，验证用户提供的credentials的有效性
        filtered_credentials = validator.validate_and_filter(credentials)

        # validate the credentials, raise exception if validation failed
        model_provider_instance.validate_provider_credentials(filtered_credentials)

        return filtered_credentials

    def model_credentials_validate(
        self, *, provider: str, model_type: ModelType, model: str, credentials: dict
    ) -> dict:
        """
        Validate model credentials

        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :param credentials: model credentials, credentials form defined in `model_credential_schema`.
        :return:
        """
        # cdg:实现思路与provider_credentials_validate一样，先获取model_provider_instance，
        # 再从model_provider_instance获取provider_schema，再从provider_schema获取model_credential_schema
        # 然后根据model_credential_schema对用户输入的credentials进行验证和过滤
        # 最后根据指定的模型实例，对模型的credentials进行验证

        # get the provider instance
        model_provider_instance = self.get_provider_instance(provider)

        # get provider schema
        provider_schema = model_provider_instance.get_provider_schema()

        # get model_credential_schema and validate credentials according to the rules
        model_credential_schema = provider_schema.model_credential_schema

        if not model_credential_schema:
            raise ValueError(f"Provider {provider} does not have model_credential_schema")

        # validate model credential schema
        validator = ModelCredentialSchemaValidator(model_type, model_credential_schema)
        filtered_credentials = validator.validate_and_filter(credentials)

        # get model instance of the model type
        model_instance = model_provider_instance.get_model_instance(model_type)

        # call validate_credentials method of model type to validate credentials, raise exception if validation failed
        model_instance.validate_credentials(model, filtered_credentials)

        return filtered_credentials

    def get_models(
        self,
        *,
        provider: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider_configs: Optional[list[ProviderConfig]] = None,
    ) -> list[SimpleProviderEntity]:
        """
        Get all models for given model type

        :param provider: provider name
        :param model_type: model type
        :param provider_configs: list of provider configs
        :return: list of models
        """
        # cdg:供应商配置信息，包括供应商名称、鉴权信息（如base_url、api-Key等）
        provider_configs = provider_configs or []

        # scan all providers
        # cdg:所有供应商字典{provider:model_provider_extensions}
        model_provider_extensions = self._get_model_provider_map()

        # cdg:将ProviderConfig实例转为字典
        # convert provider_configs to dict
        provider_credentials_dict = {}
        for provider_config in provider_configs:
            provider_credentials_dict[provider_config.provider] = provider_config.credentials

        # traverse all model_provider_extensions
        providers = []
        for name, model_provider_extension in model_provider_extensions.items():
            # filter by provider if provider is present
            # cdg:如果指定了供应商，不是指定的供应商直接跳过
            if provider and name != provider:
                continue

            # get model_provider instance
            model_provider_instance = model_provider_extension.provider_instance

            # get provider schema
            provider_schema = model_provider_instance.get_provider_schema()

            # cdg:获取供应商支持的模型类型，如llm、Embedding等
            model_types = provider_schema.supported_model_types
            if model_type:
                # cdg:如果指定了模型类型且不是指定的模型类型，则直接跳过
                if model_type not in model_types:
                    continue

                model_types = [model_type]

            # cdg: 指定供应商名称和模型类型的预定义模型列表
            all_model_type_models = []
            for model_type in model_types:
                # get predefined models for given model type
                models = model_provider_instance.models(
                    model_type=model_type,
                )


                all_model_type_models.extend(models)

            simple_provider_schema = provider_schema.to_simple_provider()
            simple_provider_schema.models.extend(all_model_type_models)

            providers.append(simple_provider_schema)

        return providers

    def get_provider_instance(self, provider: str) -> ModelProvider:
        """
        Get provider instance by provider name
        :param provider: provider name
        :return: provider instance
        """
        # scan all providers
        # cdg:供应商字典{provider:model_provider_extensions,……}
        model_provider_extensions = self._get_model_provider_map()

        # get the provider extension
        model_provider_extension = model_provider_extensions.get(provider)
        if not model_provider_extension:
            raise Exception(f"Invalid provider: {provider}")

        # get the provider instance
        model_provider_instance = model_provider_extension.provider_instance

        return model_provider_instance

    def _get_model_provider_map(self) -> dict[str, ModelProviderExtension]:
        """
        Retrieves the model provider map.

        This method retrieves the model provider map, which is a dictionary containing the model provider names as keys
        and instances of `ModelProviderExtension` as values. The model provider map is used to store information about
        available model providers.

        Returns:
            A dictionary containing the model provider map.

        Raises:
            None.
        """
        if self.model_provider_extensions:
            return self.model_provider_extensions

        # get the path of current classes
        # cdg:当前python代码绝对路径，如api/core/model_runtime/model_providers/model_provider_factory.py
        current_path = os.path.abspath(__file__)
        # cdg:当前python代码所在目录，如api/core/model_runtime/model_providers/
        model_providers_path = os.path.dirname(current_path)

        # get all folders path under model_providers_path that do not start with __
        # cdg:获取所有供应商目录，如[api/core/model_runtime/model_providers/anthropic, api/core/model_runtime/model_providers/openai, ……]
        model_provider_dir_paths = [
            os.path.join(model_providers_path, model_provider_dir)
            for model_provider_dir in os.listdir(model_providers_path)
            if not model_provider_dir.startswith("__")
            and os.path.isdir(os.path.join(model_providers_path, model_provider_dir))
        ]

        # get _position.yaml file path
        # cdg:{name:index,……}
        position_map = get_provider_position_map(model_providers_path)

        # traverse all model_provider_dir_paths
        model_providers: list[ModelProviderExtension] = []
        for model_provider_dir_path in model_provider_dir_paths:
            # get model_provider dir name
            # cdg:获取供应商名称，如anthropic、openai等
            model_provider_name = os.path.basename(model_provider_dir_path)

            # cdg:供应商目录下所有文件
            file_names = os.listdir(model_provider_dir_path)

            if (model_provider_name + ".py") not in file_names:
                logger.warning(f"Missing {model_provider_name}.py file in {model_provider_dir_path}, Skip.")
                continue

            # Dynamic loading {model_provider_name}.py file and find the subclass of ModelProvider
            # cdg:供应商目录下同名python代码文件，如api/core/model_runtime/model_providers/anthropic/anthropic.py
            py_path = os.path.join(model_provider_dir_path, model_provider_name + ".py")
            # cdg:加载供应商模型类名称，如AnthropicProvider
            model_provider_class = load_single_subclass_from_source(
                module_name=f"core.model_runtime.model_providers.{model_provider_name}.{model_provider_name}",
                script_path=py_path,
                parent_type=ModelProvider,
            )

            if not model_provider_class:
                logger.warning(f"Missing Model Provider Class that extends ModelProvider in {py_path}, Skip.")
                continue
            # cdg:缺乏.py文件，无法获取模型类名，直接报错；缺乏.yaml文件，还可以初始化，但缺乏参数
            if f"{model_provider_name}.yaml" not in file_names:
                logger.warning(f"Missing {model_provider_name}.yaml file in {model_provider_dir_path}, Skip.")
                continue

            model_providers.append(
                ModelProviderExtension(
                    name=model_provider_name,
                    provider_instance=model_provider_class(),
                    position=position_map.get(model_provider_name),
                )
            )
        
        # cdg:根据api/core/model_runtime/model_providers/_position.yaml中的顺序进行排序
        sorted_extensions = sort_to_dict_by_position_map(position_map, model_providers, lambda x: x.name)

        self.model_provider_extensions = sorted_extensions

        return sorted_extensions
