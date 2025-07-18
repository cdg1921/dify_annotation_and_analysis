from core.tools.entities.tool_entities import ToolLabel
from core.tools.entities.values import default_tool_labels

# cdg: 工具标签服务
class ToolLabelsService:
    # cdg: 列出工具标签
    @classmethod
    def list_tool_labels(cls) -> list[ToolLabel]:
        return default_tool_labels
