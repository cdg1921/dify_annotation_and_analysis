import copy

from core.prompt.prompt_templates.advanced_prompt_templates import (
    BAICHUAN_CHAT_APP_CHAT_PROMPT_CONFIG,
    BAICHUAN_CHAT_APP_COMPLETION_PROMPT_CONFIG,
    BAICHUAN_COMPLETION_APP_CHAT_PROMPT_CONFIG,
    BAICHUAN_COMPLETION_APP_COMPLETION_PROMPT_CONFIG,
    BAICHUAN_CONTEXT,
    CHAT_APP_CHAT_PROMPT_CONFIG,
    CHAT_APP_COMPLETION_PROMPT_CONFIG,
    COMPLETION_APP_CHAT_PROMPT_CONFIG,
    COMPLETION_APP_COMPLETION_PROMPT_CONFIG,
    CONTEXT,
)
from models.model import AppMode

# cdg: 高级提示模板服务，包括获取提示模板、获取通用提示模板、获取完成提示模板、获取聊天提示模板、获取白川提示模板。
class AdvancedPromptTemplateService:

    # cdg: 获取提示模板，具体实现为：根据模型名称判断是否为白川模型；如果是白川模型，则获取白川提示模板；否则获取通用提示模板。
    @classmethod
    def get_prompt(cls, args: dict) -> dict:
        app_mode = args["app_mode"]
        model_mode = args["model_mode"]
        model_name = args["model_name"]
        has_context = args["has_context"]

        if "baichuan" in model_name.lower():
            return cls.get_baichuan_prompt(app_mode, model_mode, has_context)
        else:
            return cls.get_common_prompt(app_mode, model_mode, has_context)

    # cdg: 获取通用提示模板，具体实现为：根据应用模式和模型模式获取提示模板；如果应用模式为聊天，则获取聊天提示模板；如果应用模式为完成，则获取完成提示模板。
    @classmethod
    def get_common_prompt(cls, app_mode: str, model_mode: str, has_context: str) -> dict:
        context_prompt = copy.deepcopy(CONTEXT)

        # cdg: 如果应用模式为聊天，则获取聊天提示模板；如果应用模式为补全，则获取补全提示模板。否则返回空字典。
        if app_mode == AppMode.CHAT.value:
            if model_mode == "completion":
                return cls.get_completion_prompt(
                    copy.deepcopy(CHAT_APP_COMPLETION_PROMPT_CONFIG), has_context, context_prompt
                )
            elif model_mode == "chat":
                return cls.get_chat_prompt(copy.deepcopy(CHAT_APP_CHAT_PROMPT_CONFIG), has_context, context_prompt)
        elif app_mode == AppMode.COMPLETION.value:
            if model_mode == "completion":
                return cls.get_completion_prompt(
                    copy.deepcopy(COMPLETION_APP_COMPLETION_PROMPT_CONFIG), has_context, context_prompt
                )
            elif model_mode == "chat":
                return cls.get_chat_prompt(
                    copy.deepcopy(COMPLETION_APP_CHAT_PROMPT_CONFIG), has_context, context_prompt
                )
        # default return empty dict
        return {}

    # cdg: 获取完成提示模板，具体实现为：如果上下文存在，则将上下文添加到提示模板中。
    @classmethod
    def get_completion_prompt(cls, prompt_template: dict, has_context: str, context: str) -> dict:
        if has_context == "true":
            prompt_template["completion_prompt_config"]["prompt"]["text"] = (
                context + prompt_template["completion_prompt_config"]["prompt"]["text"]
            )

        return prompt_template

    # cdg: 获取聊天提示模板，具体实现为：如果上下文存在，则将上下文添加到提示模板中。
    @classmethod
    def get_chat_prompt(cls, prompt_template: dict, has_context: str, context: str) -> dict:
        if has_context == "true":
            prompt_template["chat_prompt_config"]["prompt"][0]["text"] = (
                context + prompt_template["chat_prompt_config"]["prompt"][0]["text"]
            )

        return prompt_template

    # cdg: 获取百川提示模板，具体实现为：根据应用模式和模型模式获取提示模板；如果应用模式为聊天，则获取聊天提示模板；如果应用模式为补全，则获取补全提示模板。否则返回空字典。
    @classmethod
    def get_baichuan_prompt(cls, app_mode: str, model_mode: str, has_context: str) -> dict:
        baichuan_context_prompt = copy.deepcopy(BAICHUAN_CONTEXT)

        if app_mode == AppMode.CHAT.value:
            if model_mode == "completion":
                return cls.get_completion_prompt(
                    copy.deepcopy(BAICHUAN_CHAT_APP_COMPLETION_PROMPT_CONFIG), has_context, baichuan_context_prompt
                )
            elif model_mode == "chat":
                return cls.get_chat_prompt(
                    copy.deepcopy(BAICHUAN_CHAT_APP_CHAT_PROMPT_CONFIG), has_context, baichuan_context_prompt
                )
        elif app_mode == AppMode.COMPLETION.value:
            if model_mode == "completion":
                return cls.get_completion_prompt(
                    copy.deepcopy(BAICHUAN_COMPLETION_APP_COMPLETION_PROMPT_CONFIG),
                    has_context,
                    baichuan_context_prompt,
                )
            elif model_mode == "chat":
                return cls.get_chat_prompt(
                    copy.deepcopy(BAICHUAN_COMPLETION_APP_CHAT_PROMPT_CONFIG), has_context, baichuan_context_prompt
                )
        # default return empty dict
        return {}
