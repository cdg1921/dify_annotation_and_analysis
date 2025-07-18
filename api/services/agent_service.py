from typing import Optional

import pytz
from flask_login import current_user  # type: ignore

from core.app.app_config.easy_ui_based_app.agent.manager import AgentConfigManager
from core.tools.tool_manager import ToolManager
from extensions.ext_database import db
from models.account import Account
from models.model import App, Conversation, EndUser, Message, MessageAgentThought

# cdg: Agent服务，包括获取代理日志。 
class AgentService:
    # cdg: 获取Agent日志，具体实现为：从数据库中获取会话；如果会话不存在，则抛出异常；从数据库中获取消息；如果消息不存在，则抛出异常；获取代理思考；如果会话的执行者为终端用户，则获取终端用户；如果会话的执行者为账户，则获取账户；获取时区；构建结果。
    @classmethod
    def get_agent_logs(cls, app_model: App, conversation_id: str, message_id: str) -> dict:
        """
        Service to get agent logs
        """
        # cdg: 从数据库中获取会话
        conversation: Optional[Conversation] = (
            db.session.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.app_id == app_model.id,
            )
            .first()
        )
        # cdg: 如果会话不存在，则抛出异常
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        # cdg: 从数据库中获取消息
        message: Optional[Message] = (
            db.session.query(Message)
            .filter(
                Message.id == message_id,
                Message.conversation_id == conversation_id,
            )
            .first()
        )
        # cdg: 如果消息不存在，则抛出异常
        if not message:
            raise ValueError(f"Message not found: {message_id}")

        # cdg: 获取Agent思考过程信息
        agent_thoughts: list[MessageAgentThought] = message.agent_thoughts

        # cdg: 如果会话的执行者为终端用户，则获取终端用户；如果会话的执行者为账户，则获取账户。
        if conversation.from_end_user_id:
            # only select name field
            executor = (
                db.session.query(EndUser, EndUser.name).filter(EndUser.id == conversation.from_end_user_id).first()
            )
        else:
            executor = (
                db.session.query(Account, Account.name).filter(Account.id == conversation.from_account_id).first()
            )

        # cdg: 如果执行者存在，则获取执行者名称；否则设置为未知。
        if executor:
            executor = executor.name
        else:
            executor = "Unknown"

        # cdg: 获取时区
        timezone = pytz.timezone(current_user.timezone)

        # cdg: 构建结果
        result = {
            "meta": {
                "status": "success",
                "executor": executor,
                "start_time": message.created_at.astimezone(timezone).isoformat(),
                "elapsed_time": message.provider_response_latency,
                "total_tokens": message.answer_tokens + message.message_tokens,
                "agent_mode": app_model.app_model_config.agent_mode_dict.get("strategy", "react"),
                "iterations": len(agent_thoughts),
            },
            "iterations": [],
            "files": message.message_files,
        }
    
        # cdg: 获取Agent配置
        agent_config = AgentConfigManager.convert(app_model.app_model_config.to_dict())
        if not agent_config:
            return result

        # cdg: 获取Agent工具
        agent_tools = agent_config.tools or []

        # cdg: 查找Agent工具
        def find_agent_tool(tool_name: str):
            for agent_tool in agent_tools:
                if agent_tool.tool_name == tool_name:
                    return agent_tool

        # cdg: 遍历Agent思考过程信息，其中关键操作是：获取工具；获取工具标签；获取工具元数据；获取工具输入；获取工具输出；获取工具调用。
        for agent_thought in agent_thoughts:
            tools = agent_thought.tools
            tool_labels = agent_thought.tool_labels
            tool_meta = agent_thought.tool_meta
            tool_inputs = agent_thought.tool_inputs_dict
            tool_outputs = agent_thought.tool_outputs_dict
            tool_calls = []
            for tool in tools:
                tool_name = tool
                tool_label = tool_labels.get(tool_name, tool_name)
                tool_input = tool_inputs.get(tool_name, {})
                tool_output = tool_outputs.get(tool_name, {})
                tool_meta_data = tool_meta.get(tool_name, {})
                tool_config = tool_meta_data.get("tool_config", {})
                if tool_config.get("tool_provider_type", "") != "dataset-retrieval":
                    tool_icon = ToolManager.get_tool_icon(
                        tenant_id=app_model.tenant_id,
                        provider_type=tool_config.get("tool_provider_type", ""),
                        provider_id=tool_config.get("tool_provider", ""),
                    )
                    if not tool_icon:
                        tool_entity = find_agent_tool(tool_name)
                        if tool_entity:
                            tool_icon = ToolManager.get_tool_icon(
                                tenant_id=app_model.tenant_id,
                                provider_type=tool_entity.provider_type,
                                provider_id=tool_entity.provider_id,
                            )
                else:
                    tool_icon = ""

                # cdg: 构建工具调用
                tool_calls.append(
                    {
                        "status": "success" if not tool_meta_data.get("error") else "error",
                        "error": tool_meta_data.get("error"),
                        "time_cost": tool_meta_data.get("time_cost", 0),
                        "tool_name": tool_name,
                        "tool_label": tool_label,
                        "tool_input": tool_input,
                        "tool_output": tool_output,
                        "tool_parameters": tool_meta_data.get("tool_parameters", {}),
                        "tool_icon": tool_icon,
                    }
                )

            # cdg: 构建迭代结果
            result["iterations"].append(
                {
                    "tokens": agent_thought.tokens,
                    "tool_calls": tool_calls,
                    "tool_raw": {
                        "inputs": agent_thought.tool_input,
                        "outputs": agent_thought.observation,
                    },
                    "thought": agent_thought.thought,
                    "created_at": agent_thought.created_at.isoformat(),
                    "files": agent_thought.files,
                }
            )

        return result
