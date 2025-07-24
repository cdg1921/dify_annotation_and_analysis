import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Optional, Union, cast

import httpx
import validators

from constants import HIDDEN_VALUE
from core.helper import ssrf_proxy
from extensions.ext_database import db
from models.dataset import (
    Dataset,
    ExternalKnowledgeApis,
    ExternalKnowledgeBindings,
)
from services.entities.external_knowledge_entities.external_knowledge_entities import (
    Authorization,
    ExternalKnowledgeApiSetting,
)
from services.errors.dataset import DatasetNameDuplicateError

# cdg:外部知识库链接服务，主要用于管理和操作外部知识库API的相关功能。它定义了一个名为ExternalDatasetService的服务类，提供了丰富的方法来实现外部知识库API的增删改查、校验、调用等操作。
class ExternalDatasetService:
    # cdg:获取外部知识库API列表,返回外部知识库API列表和总数量。主要思路是：
    # 1. 构建查询条件，根据tenant_id过滤，并按创建时间降序排序
    # 2. 如果search参数不为空，则添加名称模糊搜索条件
    # 3. 使用paginate方法进行分页查询，设置最大每页数量为100，错误时输出False
    # 4. 返回查询结果的items（当前页的API列表）和total（总数量）
    @staticmethod
    def get_external_knowledge_apis(page, per_page, tenant_id, search=None) -> tuple[list[ExternalKnowledgeApis], int]:
        query = ExternalKnowledgeApis.query.filter(ExternalKnowledgeApis.tenant_id == tenant_id).order_by(
            ExternalKnowledgeApis.created_at.desc()
        )
        if search:
            query = query.filter(ExternalKnowledgeApis.name.ilike(f"%{search}%"))

        external_knowledge_apis = query.paginate(page=page, per_page=per_page, max_per_page=100, error_out=False)

        return external_knowledge_apis.items, external_knowledge_apis.total

    # cdg:验证API列表是否有效，主要用于检查API配置是否包含必要的endpoint和api_key。如果配置为空或缺少必要字段，则抛出ValueError异常。
    @classmethod
    def validate_api_list(cls, api_settings: dict):
        if not api_settings:
            raise ValueError("api list is empty")
        if "endpoint" not in api_settings and not api_settings["endpoint"]:
            raise ValueError("endpoint is required")
        if "api_key" not in api_settings and not api_settings["api_key"]:
            raise ValueError("api_key is required")

    # cdg:创建外部知识库API，主要用于创建新的外部知识库API配置。它首先验证API配置是否包含必要的endpoint和api_key，然后创建新的ExternalKnowledgeApis对象并将其添加到数据库中。
    @staticmethod
    def create_external_knowledge_api(tenant_id: str, user_id: str, args: dict) -> ExternalKnowledgeApis:
        settings = args.get("settings")
        if settings is None:
            raise ValueError("settings is required")
        ExternalDatasetService.check_endpoint_and_api_key(settings)
        external_knowledge_api = ExternalKnowledgeApis(
            tenant_id=tenant_id,
            created_by=user_id,
            updated_by=user_id,
            name=args.get("name"),
            description=args.get("description", ""),
            settings=json.dumps(args.get("settings"), ensure_ascii=False),
        )

        db.session.add(external_knowledge_api)
        db.session.commit()
        return external_knowledge_api

    # cdg:检查API配置是否包含必要的endpoint和api_key。如果配置为空或缺少必要字段，则抛出ValueError异常。
    @staticmethod
    def check_endpoint_and_api_key(settings: dict):
        if "endpoint" not in settings or not settings["endpoint"]:
            raise ValueError("endpoint is required")
        if "api_key" not in settings or not settings["api_key"]:
            raise ValueError("api_key is required")

        endpoint = f"{settings['endpoint']}/retrieval"
        api_key = settings["api_key"]
        if not validators.url(endpoint, simple_host=True):
            if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
                raise ValueError(f"invalid endpoint: {endpoint} must start with http:// or https://")
            else:
                raise ValueError(f"invalid endpoint: {endpoint}")
        try:
            response = httpx.post(endpoint, headers={"Authorization": f"Bearer {api_key}"})
        except Exception as e:
            raise ValueError(f"failed to connect to the endpoint: {endpoint}")
        if response.status_code == 502:
            raise ValueError(f"Bad Gateway: failed to connect to the endpoint: {endpoint}")
        if response.status_code == 404:
            raise ValueError(f"Not Found: failed to connect to the endpoint: {endpoint}")
        if response.status_code == 403:
            raise ValueError(f"Forbidden: Authorization failed with api_key: {api_key}")

    # cdg:获取外部知识库API，主要用于根据API ID获取对应的外部知识库API配置。它首先构建查询条件，然后执行查询操作，如果未找到对应的API配置，则抛出ValueError异常。
    @staticmethod
    def get_external_knowledge_api(external_knowledge_api_id: str) -> ExternalKnowledgeApis:
        external_knowledge_api: Optional[ExternalKnowledgeApis] = ExternalKnowledgeApis.query.filter_by(
            id=external_knowledge_api_id
        ).first()
        if external_knowledge_api is None:
            raise ValueError("api template not found")
        return external_knowledge_api

    # cdg:更新外部知识库API，主要用于更新现有外部知识库API的配置。它首先验证API配置是否包含必要的endpoint和api_key，然后更新ExternalKnowledgeApis对象的属性，并将其保存到数据库中。
    @staticmethod
    def update_external_knowledge_api(tenant_id, user_id, external_knowledge_api_id, args) -> ExternalKnowledgeApis:
        external_knowledge_api: Optional[ExternalKnowledgeApis] = ExternalKnowledgeApis.query.filter_by(
            id=external_knowledge_api_id, tenant_id=tenant_id
        ).first()
        if external_knowledge_api is None:
            raise ValueError("api template not found")
        if args.get("settings") and args.get("settings").get("api_key") == HIDDEN_VALUE:
            args.get("settings")["api_key"] = external_knowledge_api.settings_dict.get("api_key")

        external_knowledge_api.name = args.get("name")
        external_knowledge_api.description = args.get("description", "")
        external_knowledge_api.settings = json.dumps(args.get("settings"), ensure_ascii=False)
        external_knowledge_api.updated_by = user_id
        external_knowledge_api.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.session.commit()

        return external_knowledge_api

    # cdg:删除外部知识库API，主要用于删除指定ID的外部知识库API配置。它首先构建查询条件，然后执行删除操作，如果未找到对应的API配置，则抛出ValueError异常。
    @staticmethod
    def delete_external_knowledge_api(tenant_id: str, external_knowledge_api_id: str):
        external_knowledge_api = ExternalKnowledgeApis.query.filter_by(
            id=external_knowledge_api_id, tenant_id=tenant_id
        ).first()
        if external_knowledge_api is None:
            raise ValueError("api template not found")

        db.session.delete(external_knowledge_api)
        db.session.commit()

    # cdg:检查外部知识库API是否被使用，主要用于检查指定ID的外部知识库API是否被其他对象引用。它首先构建查询条件，然后执行计数操作，如果计数大于0，则返回True和计数，否则返回False和0。
    @staticmethod
    def external_knowledge_api_use_check(external_knowledge_api_id: str) -> tuple[bool, int]:
        count = ExternalKnowledgeBindings.query.filter_by(external_knowledge_api_id=external_knowledge_api_id).count()
        if count > 0:
            return True, count
        return False, 0

    # cdg:获取外部知识库绑定，主要用于根据数据集ID获取对应的外部知识库绑定配置。它首先构建查询条件，然后执行查询操作，如果未找到对应的绑定配置，则抛出ValueError异常。
    @staticmethod
    def get_external_knowledge_binding_with_dataset_id(tenant_id: str, dataset_id: str) -> ExternalKnowledgeBindings:
        external_knowledge_binding: Optional[ExternalKnowledgeBindings] = ExternalKnowledgeBindings.query.filter_by(
            dataset_id=dataset_id, tenant_id=tenant_id
        ).first()
        if not external_knowledge_binding:
            raise ValueError("external knowledge binding not found")
        return external_knowledge_binding

    # cdg:验证文档创建参数，主要用于验证文档创建参数是否符合API配置的要求。它首先构建查询条件，然后执行查询操作，如果未找到对应的API配置，则抛出ValueError异常。
    @staticmethod
    def document_create_args_validate(tenant_id: str, external_knowledge_api_id: str, process_parameter: dict):
        external_knowledge_api = ExternalKnowledgeApis.query.filter_by(
            id=external_knowledge_api_id, tenant_id=tenant_id
        ).first()
        if external_knowledge_api is None:
            raise ValueError("api template not found")
        settings = json.loads(external_knowledge_api.settings)
        for setting in settings:
            custom_parameters = setting.get("document_process_setting")
            if custom_parameters:
                for parameter in custom_parameters:
                    if parameter.get("required", False) and not process_parameter.get(parameter.get("name")):
                        raise ValueError(f'{parameter.get("name")} is required')

    # cdg:处理外部API请求，主要用于处理外部API请求。它首先构建查询条件，然后执行查询操作，如果未找到对应的API配置，则抛出ValueError异常。
    @staticmethod
    def process_external_api(
        settings: ExternalKnowledgeApiSetting, files: Union[None, dict[str, Any]]
    ) -> httpx.Response:
        """
        do http request depending on api bundle
        """

        kwargs = {
            "url": settings.url,
            "headers": settings.headers,
            "follow_redirects": True,
        }

        response: httpx.Response = getattr(ssrf_proxy, settings.request_method)(
            data=json.dumps(settings.params), files=files, **kwargs
        )
        return response

    # cdg:组装请求头，主要用于组装请求头。它首先构建查询条件，然后执行查询操作，如果未找到对应的API配置，则抛出ValueError异常。
    @staticmethod
    def assembling_headers(authorization: Authorization, headers: Optional[dict] = None) -> dict[str, Any]:
        authorization = deepcopy(authorization)
        if headers:
            headers = deepcopy(headers)
        else:
            headers = {}
        if authorization.type == "api-key":
            if authorization.config is None:
                raise ValueError("authorization config is required")

            if authorization.config.api_key is None:
                raise ValueError("api_key is required")

            if not authorization.config.header:
                authorization.config.header = "Authorization"

            if authorization.config.type == "bearer":
                headers[authorization.config.header] = f"Bearer {authorization.config.api_key}"
            elif authorization.config.type == "basic":
                headers[authorization.config.header] = f"Basic {authorization.config.api_key}"
            elif authorization.config.type == "custom":
                headers[authorization.config.header] = authorization.config.api_key

        return headers

    # cdg:获取外部知识库API设置，主要用于将字典转换为ExternalKnowledgeApiSetting对象。它使用ExternalKnowledgeApiSetting.parse_obj方法将字典转换为对象。
    @staticmethod
    def get_external_knowledge_api_settings(settings: dict) -> ExternalKnowledgeApiSetting:
        return ExternalKnowledgeApiSetting.parse_obj(settings)

    # cdg:创建外部数据集，主要用于创建新的外部数据集。它首先检查数据集名称是否已存在，然后获取外部知识库API配置，并创建新的Dataset对象。最后，它创建ExternalKnowledgeBindings对象，并将其添加到数据库中。
    @staticmethod
    def create_external_dataset(tenant_id: str, user_id: str, args: dict) -> Dataset:
        # check if dataset name already exists
        if Dataset.query.filter_by(name=args.get("name"), tenant_id=tenant_id).first():
            raise DatasetNameDuplicateError(f"Dataset with name {args.get('name')} already exists.")
        external_knowledge_api = ExternalKnowledgeApis.query.filter_by(
            id=args.get("external_knowledge_api_id"), tenant_id=tenant_id
        ).first()

        if external_knowledge_api is None:
            raise ValueError("api template not found")

        dataset = Dataset(
            tenant_id=tenant_id,
            name=args.get("name"),
            description=args.get("description", ""),
            provider="external",
            retrieval_model=args.get("external_retrieval_model"),
            created_by=user_id,
        )

        db.session.add(dataset)
        db.session.flush()

        external_knowledge_binding = ExternalKnowledgeBindings(
            tenant_id=tenant_id,
            dataset_id=dataset.id,
            external_knowledge_api_id=args.get("external_knowledge_api_id"),
            external_knowledge_id=args.get("external_knowledge_id"),
            created_by=user_id,
        )
        db.session.add(external_knowledge_binding)

        db.session.commit()

        return dataset

    # cdg:获取外部知识库检索，主要用于根据数据集ID、查询语句和检索参数获取外部知识库检索结果。它首先获取外部知识库绑定配置，然后获取外部知识库API配置，并组装请求参数。最后，它调用process_external_api方法处理外部API请求，并返回检索结果。
    @staticmethod
    def fetch_external_knowledge_retrieval(
        tenant_id: str, dataset_id: str, query: str, external_retrieval_parameters: dict
    ) -> list:
        external_knowledge_binding = ExternalKnowledgeBindings.query.filter_by(
            dataset_id=dataset_id, tenant_id=tenant_id
        ).first()
        if not external_knowledge_binding:
            raise ValueError("external knowledge binding not found")

        external_knowledge_api = ExternalKnowledgeApis.query.filter_by(
            id=external_knowledge_binding.external_knowledge_api_id
        ).first()
        if not external_knowledge_api:
            raise ValueError("external api template not found")

        settings = json.loads(external_knowledge_api.settings)
        headers = {"Content-Type": "application/json"}
        if settings.get("api_key"):
            headers["Authorization"] = f"Bearer {settings.get('api_key')}"
        score_threshold_enabled = external_retrieval_parameters.get("score_threshold_enabled") or False
        score_threshold = external_retrieval_parameters.get("score_threshold", 0.0) if score_threshold_enabled else 0.0
        request_params = {
            "retrieval_setting": {
                "top_k": external_retrieval_parameters.get("top_k"),
                "score_threshold": score_threshold,
            },
            "query": query,
            "knowledge_id": external_knowledge_binding.external_knowledge_id,
        }

        response = ExternalDatasetService.process_external_api(
            ExternalKnowledgeApiSetting(
                url=f"{settings.get('endpoint')}/retrieval",
                request_method="post",
                headers=headers,
                params=request_params,
            ),
            None,
        )
        if response.status_code == 200:
            return cast(list[Any], response.json().get("records", []))
        return []
