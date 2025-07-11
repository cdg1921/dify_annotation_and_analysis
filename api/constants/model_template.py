import json
from collections.abc import Mapping

from models.model import AppMode

# cdg: 应用模板默认配置信息，包括workflow、completion、chat、advanced-chat、agent-chat等，用于APP的创建
default_app_templates: Mapping[AppMode, Mapping] = {
    # workflow default mode
    # cdg: 工作流默认模式
    AppMode.WORKFLOW: {
        "app": {
            "mode": AppMode.WORKFLOW.value,
            "enable_site": True,
            "enable_api": True,
        }
    },
    # cdg: 对话补全默认模式
    # completion default mode
    AppMode.COMPLETION: {
        "app": {
            "mode": AppMode.COMPLETION.value,
            "enable_site": True,
            "enable_api": True,
        },
        "model_config": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "mode": "chat",
                "completion_params": {},
            },
            "user_input_form": json.dumps(
                [
                    {
                        "paragraph": {
                            "label": "Query",
                            "variable": "query",
                            "required": True,
                            "default": "",
                        },
                    },
                ]
            ),
            "pre_prompt": "{{query}}",
        },
    },
    # cdg: 工作流对话默认模式
    # chat default mode
    AppMode.CHAT: {
        "app": {
            "mode": AppMode.CHAT.value,
            "enable_site": True,
            "enable_api": True,
        },
        "model_config": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "mode": "chat",
                "completion_params": {},
            },
        },
    },
    # cdg: 高级对话（聊天助手）默认模式（高级对话模式是基于对话模式的基础上，增加了一些高级功能，如多轮对话、上下文管理、对话历史记录等）
    # advanced-chat default mode
    AppMode.ADVANCED_CHAT: {
        "app": {
            "mode": AppMode.ADVANCED_CHAT.value,
            "enable_site": True,
            "enable_api": True,
        },
    },
    # cdg: 智能体对话默认模式
    # agent-chat default mode
    AppMode.AGENT_CHAT: {
        "app": {
            "mode": AppMode.AGENT_CHAT.value,
            "enable_site": True,
            "enable_api": True,
        },
        "model_config": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "mode": "chat",
                "completion_params": {},
            },
        },
    },
}
