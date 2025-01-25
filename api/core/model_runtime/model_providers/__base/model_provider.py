import os
from abc import ABC, abstractmethod
from typing import Optional

from core.helper.module_import_helper import get_subclasses_from_module, import_module_from_source
from core.model_runtime.entities.model_entities import AIModelEntity, ModelType
from core.model_runtime.entities.provider_entities import ProviderEntity
from core.model_runtime.model_providers.__base.ai_model import AIModel
from core.tools.utils.yaml_utils import load_yaml_file

# cdg:供应商及其模型实例化的方式不同于工具实例化的方式，前者读取python文件和yaml文件进行实例化，后端采用mapping的方式
class ModelProvider(ABC):
    provider_schema: Optional[ProviderEntity] = None
    model_instance_map: dict[str, AIModel] = {}

    # cdg:AIModel -> AIModelEntity(添加模型参数) -> ProviderModel(模型名称、特性、配置等)
    # 此外，作为ModelProviderFactory的主要组成部分。

    @abstractmethod
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider credentials
        You can choose any validate_credentials method of model type or implement validate method by yourself,
        such as: get model list api

        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        raise NotImplementedError

    def get_provider_schema(self) -> ProviderEntity:
        """
        Get provider schema

        :return: provider schema
        """
        # cdg:如果不为空，则直接返回，否则从指定的yaml文件中解析
        if self.provider_schema:
            return self.provider_schema

        # cdg:获取当前对象所属类的模块名称的最后一部分作为供应商名称
        # get dirname of the current path
        provider_name = self.__class__.__module__.split(".")[-1]

        # cdg:获取当前python脚本文件路径，api/core/model_runtime/model_providers/__base/model_provider.py
        # get the path of the model_provider classes
        base_path = os.path.abspath(__file__)
        # cdg:api/core/model_runtime/model_providers/
        current_path = os.path.join(os.path.dirname(os.path.dirname(base_path)), provider_name)

        # cdg:从供应商目录下的yaml文件读取供应商配置
        # read provider schema from yaml file
        yaml_path = os.path.join(current_path, f"{provider_name}.yaml")
        yaml_data = load_yaml_file(yaml_path)

        try:
            # cdg:将yaml文件配置信息填充到ProviderEntity对ProviderEntity进行实例化
            # yaml_data to entity
            provider_schema = ProviderEntity(**yaml_data)
        except Exception as e:
            raise Exception(f"Invalid provider schema for {provider_name}: {str(e)}")

        # cdg:将供应商provider_schema缓存到self.provider_schema中，以免重复上述操作
        # cache schema
        self.provider_schema = provider_schema

        return provider_schema

    def models(self, model_type: ModelType) -> list[AIModelEntity]:
        """
        Get all models for given model type

        :param model_type: model type defined in `ModelType`
        :return: list of models
        """
        provider_schema = self.get_provider_schema()
        # cdg:模型类型检查，如果不支持则直接返回空列表
        if model_type not in provider_schema.supported_model_types:
            return []

        # cdg:根据模型类型获取模型实例列表
        # get model instance of the model type
        model_instance = self.get_model_instance(model_type)

        # cdg:读取预定义模型列表
        # get predefined models (predefined_models)
        models = model_instance.predefined_models()

        # return models
        return models

    def get_model_instance(self, model_type: ModelType) -> AIModel:
        """
        Get model instance

        :param model_type: model type defined in `ModelType`
        :return:
        """
        # cdg:获取当前对象所属类的模块名称的最后一部分作为供应商名称
        # get dirname of the current path
        provider_name = self.__class__.__module__.split(".")[-1]

        # cdg：model_instance_map的键值是供应商名称+类型，Key是AIModel，包括model_type、AIModelEntity列表等
        if f"{provider_name}.{model_type.value}" in self.model_instance_map:
            return self.model_instance_map[f"{provider_name}.{model_type.value}"]

        # cdg:获取当前python脚本文件路径，api/core/model_runtime/model_providers/__base/model_provider.py
        # get the path of the model type classes
        base_path = os.path.abspath(__file__)
        model_type_name = model_type.value.replace("-", "_")
        # cdg:model_type_path为：api/core/model_runtime/model_providers/
        model_type_path = os.path.join(os.path.dirname(os.path.dirname(base_path)), provider_name, model_type_name)
        model_type_py_path = os.path.join(model_type_path, f"{model_type_name}.py")

        if not os.path.isdir(model_type_path) or not os.path.exists(model_type_py_path):
            raise Exception(f"Invalid model type {model_type} for provider {provider_name}")

        # cdg:对象所属类的模块的父模块
        # Dynamic loading {model_type_name}.py file and find the subclass of AIModel
        parent_module = ".".join(self.__class__.__module__.split(".")[:-1])
        # cdg:从python代码中导入类模块
        mod = import_module_from_source(
            module_name=f"{parent_module}.{model_type_name}.{model_type_name}", py_file_path=model_type_py_path
        )
        # FIXME "type" has no attribute "__abstractmethods__" ignore it for now fix it later
        # cdg:找到指定模型的类名，为LargeLanguageModel、ModerationModel、RerankModel、Speech2TextModel、TextEmbeddingModel、Text2ImageModel、TTSModel的子类
        # 如AnthropicLargeLanguageModel
        model_class = next(
            filter(
                lambda x: x.__module__ == mod.__name__ and not x.__abstractmethods__,  # type: ignore
                get_subclasses_from_module(mod, AIModel),
            ),
            None, # cdg:next迭代完找不到则返回None值
        )
        if not model_class:
            raise Exception(f"Missing AIModel Class for model type {model_type} in {model_type_py_path}")

        # cdg:根据类名实例化AI模型对象，如model_instance_map=AnthropicLargeLanguageModel()
        model_instance_map = model_class()

        # cdg:将AI模型实例映射到字典中,如“{anthropic.llm:AnthropicLargeLanguageModel()}”
        self.model_instance_map[f"{provider_name}.{model_type.value}"] = model_instance_map

        return model_instance_map
