from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.workflow.entities.variable_pool import VariablePool

# cdg: 租户ID（用于标识当前工作流所属的租户），用于隔离不同租户之间的数据，避免数据泄露和冲突。
tenant_id: ContextVar[str] = ContextVar("tenant_id")
# cdg: 工作流变量池（用于存储工作流中的会话变量，如用户输入的文本、模型生成的文本、上下文信息等）
workflow_variable_pool: ContextVar["VariablePool"] = ContextVar("workflow_variable_pool")
