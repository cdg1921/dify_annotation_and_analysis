import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from configs import dify_config
from core.helper.position_helper import is_filtered
from core.model_runtime.utils.encoders import jsonable_encoder
from core.tools.entities.api_entities import UserTool, UserToolProvider
from core.tools.errors import ToolNotFoundError, ToolProviderCredentialValidationError, ToolProviderNotFoundError
from core.tools.provider.builtin._positions import BuiltinToolProviderSort
from core.tools.provider.tool_provider import ToolProviderController
from core.tools.tool_label_manager import ToolLabelManager
from core.tools.tool_manager import ToolManager
from core.tools.utils.configuration import ToolConfigurationManager
from extensions.ext_database import db
from models.tools import BuiltinToolProvider
from services.tools.tools_transform_service import ToolTransformService

logger = logging.getLogger(__name__)

# cdg: 内置工具管理服务 
class BuiltinToolManageService:
    # cdg: 列出内置工具提供者工具
    @staticmethod
    def list_builtin_tool_provider_tools(user_id: str, tenant_id: str, provider: str) -> list[UserTool]:
        """
        list builtin tool provider tools
        """
        # cdg: 获取内置工具提供者
        provider_controller: ToolProviderController = ToolManager.get_builtin_provider(provider)
        # cdg: 获取工具列表
        tools = provider_controller.get_tools()
        # cdg: 获取工具配置
        tool_provider_configurations = ToolConfigurationManager(
            tenant_id=tenant_id, provider_controller=provider_controller
        )
        # cdg: 检查用户是否添加了提供者
        builtin_provider = (
            db.session.query(BuiltinToolProvider)
            .filter(
                BuiltinToolProvider.tenant_id == tenant_id,
                BuiltinToolProvider.provider == provider,
            )
            .first()
        )

        credentials = {}
        if builtin_provider is not None:
            # get credentials
            credentials = builtin_provider.credentials
            credentials = tool_provider_configurations.decrypt_tool_credentials(credentials)

        # cdg: 转换工具为用户工具
        result: list[UserTool] = []
        for tool in tools or []:
            result.append(
                ToolTransformService.tool_to_user_tool(
                    tool=tool,
                    credentials=credentials,
                    tenant_id=tenant_id,
                    labels=ToolLabelManager.get_tool_labels(provider_controller),
                )
            )

        return result

    # cdg: 列出内置提供者凭证模式
    @staticmethod
    def list_builtin_provider_credentials_schema(provider_name):
        """
        list builtin provider credentials schema

        :return: the list of tool providers
        """
        provider = ToolManager.get_builtin_provider(provider_name)
        return jsonable_encoder([v for _, v in (provider.credentials_schema or {}).items()])

    # cdg: 更新内置工具提供者
    @staticmethod
    def update_builtin_tool_provider(
        session: Session, user_id: str, tenant_id: str, provider_name: str, credentials: dict
    ):
        """
        update builtin tool provider
        """
        # get if the provider exists
        stmt = select(BuiltinToolProvider).where(
            BuiltinToolProvider.tenant_id == tenant_id,
            BuiltinToolProvider.provider == provider_name,
        )
        provider = session.scalar(stmt)

        try:
            # get provider
            provider_controller = ToolManager.get_builtin_provider(provider_name)
            # cdg: 如果提供者不需要凭证，则抛出异常
            if not provider_controller.need_credentials:
                raise ValueError(f"provider {provider_name} does not need credentials")
            tool_configuration = ToolConfigurationManager(tenant_id=tenant_id, provider_controller=provider_controller)
            # cdg: 如果提供者存在，则获取原始凭证
            if provider is not None:
                original_credentials = tool_configuration.decrypt_tool_credentials(provider.credentials)
                masked_credentials = tool_configuration.mask_tool_credentials(original_credentials)
                # check if the credential has changed, save the original credential
                # cdg: 检查凭证是否发生变化，如果发生变化，则保存原始凭证
                for name, value in credentials.items():
                    if name in masked_credentials and value == masked_credentials[name]:
                        credentials[name] = original_credentials[name]
            # validate credentials

            provider_controller.validate_credentials(credentials)
            # encrypt credentials
            credentials = tool_configuration.encrypt_tool_credentials(credentials)
        except (ToolProviderNotFoundError, ToolNotFoundError, ToolProviderCredentialValidationError) as e:
            raise ValueError(str(e))

        # cdg: 如果提供者不存在，则创建提供者信息，并保存到数据库
        if provider is None:
            # create provider
            provider = BuiltinToolProvider(
                tenant_id=tenant_id,
                user_id=user_id,
                provider=provider_name,
                encrypted_credentials=json.dumps(credentials),
            )

            session.add(provider)

        else:
            # cdg: 如果提供者存在，则更新凭证
            provider.encrypted_credentials = json.dumps(credentials)

            # delete cache
            tool_configuration.delete_tool_credentials_cache()

        return {"result": "success"}

    # cdg: 获取内置工具提供者凭证
    @staticmethod
    def get_builtin_tool_provider_credentials(tenant_id: str, provider_name: str):
        """
        get builtin tool provider credentials
        """
        # cdg: 获取内置工具提供者
        provider = (
            db.session.query(BuiltinToolProvider)
            .filter(
                BuiltinToolProvider.tenant_id == tenant_id,
                BuiltinToolProvider.provider == provider_name,
            )
            .first()
        )

        if provider is None:
            return {}
        # cdg: 获取工具配置
        provider_controller = ToolManager.get_builtin_provider(provider.provider)
        # cdg: 获取工具配置
        tool_configuration = ToolConfigurationManager(tenant_id=tenant_id, provider_controller=provider_controller)
        # cdg: 解密凭证
        credentials = tool_configuration.decrypt_tool_credentials(provider.credentials)
        # cdg: 掩码凭证，即中间加星号
        credentials = tool_configuration.mask_tool_credentials(credentials)
        return credentials

    # cdg: 删除内置工具提供者
    @staticmethod
    def delete_builtin_tool_provider(user_id: str, tenant_id: str, provider_name: str):
        """
        delete tool provider
        """
        provider = (
            db.session.query(BuiltinToolProvider)
            .filter(
                BuiltinToolProvider.tenant_id == tenant_id,
                BuiltinToolProvider.provider == provider_name,
            )
            .first()
        )

        if provider is None:
            raise ValueError(f"you have not added provider {provider_name}")

        db.session.delete(provider)
        db.session.commit()

        # delete cache
        provider_controller = ToolManager.get_builtin_provider(provider_name)
        tool_configuration = ToolConfigurationManager(tenant_id=tenant_id, provider_controller=provider_controller)
        tool_configuration.delete_tool_credentials_cache()

        return {"result": "success"}

    # cdg: 获取内置工具提供者图标
    @staticmethod
    def get_builtin_tool_provider_icon(provider: str):
        """
        get tool provider icon and it's mimetype
        """
        icon_path, mime_type = ToolManager.get_builtin_provider_icon(provider)
        icon_bytes = Path(icon_path).read_bytes()

        return icon_bytes, mime_type

    # cdg: 列出内置工具
    @staticmethod
    def list_builtin_tools(user_id: str, tenant_id: str) -> list[UserToolProvider]:
        """
        list builtin tools
        """
        # get all builtin providers
        provider_controllers = ToolManager.list_builtin_providers()

        # get all user added providers
        db_providers: list[BuiltinToolProvider] = (
            db.session.query(BuiltinToolProvider).filter(BuiltinToolProvider.tenant_id == tenant_id).all() or []
        )

        # find provider
        find_provider = lambda provider: next(
            filter(lambda db_provider: db_provider.provider == provider, db_providers), None
        )

        result: list[UserToolProvider] = []

        for provider_controller in provider_controllers:
            try:
                # handle include, exclude
                if is_filtered(
                    include_set=dify_config.POSITION_TOOL_INCLUDES_SET,
                    exclude_set=dify_config.POSITION_TOOL_EXCLUDES_SET,
                    data=provider_controller,
                    name_func=lambda x: x.identity.name,
                ):
                    continue
                if provider_controller.identity is None:
                    continue

                # convert provider controller to user provider
                user_builtin_provider = ToolTransformService.builtin_provider_to_user_provider(
                    provider_controller=provider_controller,
                    db_provider=find_provider(provider_controller.identity.name),
                    decrypt_credentials=True,
                )

                # add icon
                ToolTransformService.repack_provider(user_builtin_provider)

                tools = provider_controller.get_tools()
                for tool in tools or []:
                    user_builtin_provider.tools.append(
                        ToolTransformService.tool_to_user_tool(
                            tenant_id=tenant_id,
                            tool=tool,
                            credentials=user_builtin_provider.original_credentials,
                            labels=ToolLabelManager.get_tool_labels(provider_controller),
                        )
                    )

                result.append(user_builtin_provider)
            except Exception as e:
                raise e

        return BuiltinToolProviderSort.sort(result)
