from core.tools.tool_manager import ToolManager
from core.tools.utils.configuration import ToolParameterConfigurationManager
from core.workflow.nodes import NodeType
from core.workflow.nodes.tool.entities import ToolEntity
from events.app_event import app_draft_workflow_was_synced

# cdg: 使用@app_draft_workflow_was_synced.connect装饰器，将handle函数注册为草稿工作流同步事件的处理器。
@app_draft_workflow_was_synced.connect
def handle(sender, **kwargs): # cdg: 处理草稿工作流同步事件，sender是应用对象，kwargs是事件参数
    app = sender # cdg: 获取应用对象
    synced_draft_workflow = kwargs.get("synced_draft_workflow") # cdg: 获取同步的草稿工作流
    if synced_draft_workflow is None: # cdg: 如果同步的草稿工作流为空，则返回
        return
    for node_data in synced_draft_workflow.graph_dict.get("nodes", []): # cdg: 遍历同步的草稿工作流的节点
        if node_data.get("data", {}).get("type") == NodeType.TOOL.value: # cdg: 如果节点类型为TOOL，则删除工具参数缓存
            try:
                tool_entity = ToolEntity(**node_data["data"]) # cdg: 获取工具实体
                tool_runtime = ToolManager.get_tool_runtime( # cdg: 获取工具运行时
                    provider_type=tool_entity.provider_type, # cdg: 获取工具提供者类型
                    provider_id=tool_entity.provider_id, # cdg: 获取工具提供者ID
                    tool_name=tool_entity.tool_name, # cdg: 获取工具名称
                    tenant_id=app.tenant_id, # cdg: 获取租户ID
                )
                manager = ToolParameterConfigurationManager( # cdg: 获取工具参数配置管理器
                    tenant_id=app.tenant_id, # cdg: 获取租户ID
                    tool_runtime=tool_runtime, # cdg: 获取工具运行时环境
                    provider_name=tool_entity.provider_name, # cdg: 获取工具提供者名称
                    provider_type=tool_entity.provider_type, # cdg: 获取工具提供者类型
                    identity_id=f'WORKFLOW.{app.id}.{node_data.get("id")}', # cdg: 获取工具标识ID
                )
                manager.delete_tool_parameters_cache() # cdg: 删除工具参数缓存
            except:
                # tool dose not exist cdg: 工具不存在
                pass
