import logging

from core.tools.entities.api_entities import UserToolProviderTypeLiteral
from core.tools.tool_manager import ToolManager
from services.tools.tools_transform_service import ToolTransformService

logger = logging.getLogger(__name__)

# cdg: 工具通用服务
class ToolCommonService:
    # cdg: 列出工具提供者
    @staticmethod
    def list_tool_providers(user_id: str, tenant_id: str, typ: UserToolProviderTypeLiteral = None):
        """
        list tool providers

        :return: the list of tool providers
        """
        providers = ToolManager.user_list_providers(user_id, tenant_id, typ)

        # cdg: 添加图标
        # add icon
        for provider in providers:
            ToolTransformService.repack_provider(provider)

        # cdg: 转换为字典
        result = [provider.to_dict() for provider in providers]

        return result
