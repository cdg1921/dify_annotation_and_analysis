from collections.abc import Mapping, Sequence
from typing import Optional, cast

from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.file import file_manager
from core.file.models import File
from core.helper.code_executor.jinja2.jinja2_formatter import Jinja2Formatter
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_runtime.entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContent,
    PromptMessageRole,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import ImagePromptMessageContent
from core.prompt.entities.advanced_prompt_entities import ChatModelMessage, CompletionModelPromptTemplate, MemoryConfig
from core.prompt.prompt_transform import PromptTransform
from core.prompt.utils.prompt_template_parser import PromptTemplateParser
from core.workflow.entities.variable_pool import VariablePool

"""
AI生成的注释：
### AdvancedPromptTransform 和 SimplePromptTransform 的区别
#### 1. 功能复杂度
- **SimplePromptTransform**：
  - 主要用于基本模式（Basic Mode），功能较为简单。
  - 主要处理两种模式：`CHAT` 和 `COMPLETION`，并根据输入参数生成相应的提示信息。
  - 提示模板的构建和变量替换逻辑相对直接，适用于大多数常规场景。

- **AdvancedPromptTransform**：
  - 通常用于高级模式（Advanced Mode），功能更为复杂和灵活。
  - 支持更多的自定义配置选项，如更复杂的提示模板结构、动态变量插入、多轮对话管理等。
  - 可能包含更复杂的逻辑来处理不同的应用场景，例如支持多种语言、上下文感知、个性化推荐等。

#### 2. 提示模板解析
- **SimplePromptTransform**：
  - 使用 `PromptTemplateParser` 解析简单的提示模板。
  - 提示模板的构建主要依赖于预定义的规则文件 (`prompt_rules`) 和用户输入的变量。
  - 变量替换逻辑较为固定，主要用于替换特定的关键字（如 `#query#`, `#context#`, `#histories#`）。

- **AdvancedPromptTransform**：
  - 可能使用更复杂的模板解析器，支持嵌套模板、条件分支、循环结构等高级特性。
  - 提示模板的构建更加灵活，可以根据不同的上下文动态调整模板内容。
  - 支持更复杂的变量替换逻辑，包括动态生成变量、基于上下文的变量选择等。

#### 3. 历史对话管理
- **SimplePromptTransform**：
  - 在 `COMPLETION` 模式下，考虑历史对话的记忆，但实现较为简单。
  - 主要通过 `TokenBufferMemory` 来管理历史对话，并在必要时将历史对话追加到提示信息中。

- **AdvancedPromptTransform**：
  - 可能包含更复杂的对话管理机制，如多轮对话跟踪、对话状态管理、对话上下文持久化等。
  - 支持更精细的历史对话控制，例如根据对话长度、时间戳、对话主题等进行筛选和过滤。

#### 4. 扩展性和灵活性
- **SimplePromptTransform**：
  - 代码结构较为紧凑，扩展性有限。
  - 主要适用于不需要复杂对话管理和高级提示定制的场景。

- **AdvancedPromptTransform**：
  - 代码结构更为模块化，具有更高的扩展性和灵活性。
  - 支持插件化设计，可以方便地添加新的功能模块或自定义逻辑。
  - 更容易集成第三方服务或工具，以增强提示生成的能力。

#### 5. 应用场景
- **SimplePromptTransform**：
  - 适用于基础的聊天机器人应用，如客服机器人、问答系统等。
  - 场景较为固定，变化较少。

- **AdvancedPromptTransform**：
  - 适用于需要高度定制化的高级应用场景，如智能助手、个性化推荐系统、多轮对话系统等。
  - 需要处理复杂的用户交互和多样化的业务需求。

### 总结
`SimplePromptTransform` 和 `AdvancedPromptTransform` 的主要区别在于功能复杂度、提示模板解析能力、历史对话管理机制、扩展性和灵活性以及适用的应用场景。`SimplePromptTransform` 适合处理简单的提示生成任务，而 `AdvancedPromptTransform` 则提供了更强大的功能和灵活性，适用于更复杂的对话管理和提示定制需求。
"""
class AdvancedPromptTransform(PromptTransform):
    """
    Advanced Prompt Transform for Workflow LLM Node.
    """

    def __init__(
        self,
        with_variable_tmpl: bool = False,
        image_detail_config: ImagePromptMessageContent.DETAIL = ImagePromptMessageContent.DETAIL.LOW,
    ) -> None:
        self.with_variable_tmpl = with_variable_tmpl
        self.image_detail_config = image_detail_config

    def get_prompt(
        self,
        *,
        prompt_template: Sequence[ChatModelMessage] | CompletionModelPromptTemplate,
        inputs: Mapping[str, str],
        query: str,
        files: Sequence[File],
        context: Optional[str],
        memory_config: Optional[MemoryConfig],
        memory: Optional[TokenBufferMemory],
        model_config: ModelConfigWithCredentialsEntity,
    ) -> list[PromptMessage]:
        prompt_messages = []

        if isinstance(prompt_template, CompletionModelPromptTemplate):
            prompt_messages = self._get_completion_model_prompt_messages(
                prompt_template=prompt_template,
                inputs=inputs,
                query=query,
                files=files,
                context=context,
                memory_config=memory_config,
                memory=memory,
                model_config=model_config,
            )
        elif isinstance(prompt_template, list) and all(isinstance(item, ChatModelMessage) for item in prompt_template):
            prompt_messages = self._get_chat_model_prompt_messages(
                prompt_template=prompt_template,
                inputs=inputs,
                query=query,
                files=files,
                context=context,
                memory_config=memory_config,
                memory=memory,
                model_config=model_config,
            )

        return prompt_messages

    def _get_completion_model_prompt_messages(
        self,
        prompt_template: CompletionModelPromptTemplate,
        inputs: Mapping[str, str],
        query: Optional[str],
        files: Sequence[File],
        context: Optional[str],
        memory_config: Optional[MemoryConfig],
        memory: Optional[TokenBufferMemory],
        model_config: ModelConfigWithCredentialsEntity,
    ) -> list[PromptMessage]:
        """
        Get completion model prompt messages.
        """
        raw_prompt = prompt_template.text

        prompt_messages: list[PromptMessage] = []

        if prompt_template.edition_type == "basic" or not prompt_template.edition_type:
            parser = PromptTemplateParser(template=raw_prompt, with_variable_tmpl=self.with_variable_tmpl)
            prompt_inputs: Mapping[str, str] = {k: inputs[k] for k in parser.variable_keys if k in inputs}

            prompt_inputs = self._set_context_variable(context, parser, prompt_inputs)

            if memory and memory_config and memory_config.role_prefix:
                role_prefix = memory_config.role_prefix
                prompt_inputs = self._set_histories_variable(
                    memory=memory,
                    memory_config=memory_config,
                    raw_prompt=raw_prompt,
                    role_prefix=role_prefix,
                    parser=parser,
                    prompt_inputs=prompt_inputs,
                    model_config=model_config,
                )

            if query:
                prompt_inputs = self._set_query_variable(query, parser, prompt_inputs)

            prompt = parser.format(prompt_inputs)
        else:
            prompt = raw_prompt
            prompt_inputs = inputs

            prompt = Jinja2Formatter.format(prompt, prompt_inputs)

        if files:
            prompt_message_contents: list[PromptMessageContent] = []
            prompt_message_contents.append(TextPromptMessageContent(data=prompt))
            for file in files:
                prompt_message_contents.append(file_manager.to_prompt_message_content(file))

            prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
        else:
            prompt_messages.append(UserPromptMessage(content=prompt))

        return prompt_messages

    def _get_chat_model_prompt_messages(
        self,
        prompt_template: list[ChatModelMessage],
        inputs: Mapping[str, str],
        query: Optional[str],
        files: Sequence[File],
        context: Optional[str],
        memory_config: Optional[MemoryConfig],
        memory: Optional[TokenBufferMemory],
        model_config: ModelConfigWithCredentialsEntity,
    ) -> list[PromptMessage]:
        """
        Get chat model prompt messages.
        """
        prompt_messages: list[PromptMessage] = []
        for prompt_item in prompt_template:
            raw_prompt = prompt_item.text

            if prompt_item.edition_type == "basic" or not prompt_item.edition_type:
                if self.with_variable_tmpl:
                    vp = VariablePool()
                    for k, v in inputs.items():
                        if k.startswith("#"):
                            vp.add(k[1:-1].split("."), v)
                    raw_prompt = raw_prompt.replace("{{#context#}}", context or "")
                    prompt = vp.convert_template(raw_prompt).text
                else:
                    parser = PromptTemplateParser(template=raw_prompt, with_variable_tmpl=self.with_variable_tmpl)
                    prompt_inputs: Mapping[str, str] = {k: inputs[k] for k in parser.variable_keys if k in inputs}
                    prompt_inputs = self._set_context_variable(
                        context=context, parser=parser, prompt_inputs=prompt_inputs
                    )
                    prompt = parser.format(prompt_inputs)
            elif prompt_item.edition_type == "jinja2":
                prompt = raw_prompt
                prompt_inputs = inputs
                prompt = Jinja2Formatter.format(template=prompt, inputs=prompt_inputs)
            else:
                raise ValueError(f"Invalid edition type: {prompt_item.edition_type}")

            if prompt_item.role == PromptMessageRole.USER:
                prompt_messages.append(UserPromptMessage(content=prompt))
            elif prompt_item.role == PromptMessageRole.SYSTEM and prompt:
                prompt_messages.append(SystemPromptMessage(content=prompt))
            elif prompt_item.role == PromptMessageRole.ASSISTANT:
                prompt_messages.append(AssistantPromptMessage(content=prompt))

        if query and memory_config and memory_config.query_prompt_template:
            parser = PromptTemplateParser(
                template=memory_config.query_prompt_template, with_variable_tmpl=self.with_variable_tmpl
            )
            prompt_inputs = {k: inputs[k] for k in parser.variable_keys if k in inputs}
            prompt_inputs["#sys.query#"] = query

            prompt_inputs = self._set_context_variable(context, parser, prompt_inputs)

            query = parser.format(prompt_inputs)

        if memory and memory_config:
            prompt_messages = self._append_chat_histories(memory, memory_config, prompt_messages, model_config)

            if files and query is not None:
                prompt_message_contents: list[PromptMessageContent] = []
                prompt_message_contents.append(TextPromptMessageContent(data=query))
                for file in files:
                    prompt_message_contents.append(file_manager.to_prompt_message_content(file))
                prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
            else:
                prompt_messages.append(UserPromptMessage(content=query))
        elif files:
            if not query:
                # get last message
                last_message = prompt_messages[-1] if prompt_messages else None
                if last_message and last_message.role == PromptMessageRole.USER:
                    # get last user message content and add files
                    prompt_message_contents = [TextPromptMessageContent(data=cast(str, last_message.content))]
                    for file in files:
                        prompt_message_contents.append(file_manager.to_prompt_message_content(file))

                    last_message.content = prompt_message_contents
                else:
                    prompt_message_contents = [TextPromptMessageContent(data="")]  # not for query
                    for file in files:
                        prompt_message_contents.append(file_manager.to_prompt_message_content(file))

                    prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
            else:
                prompt_message_contents = [TextPromptMessageContent(data=query)]
                for file in files:
                    prompt_message_contents.append(file_manager.to_prompt_message_content(file))

                prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
        elif query:
            prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _set_context_variable(
        self, context: str | None, parser: PromptTemplateParser, prompt_inputs: Mapping[str, str]
    ) -> Mapping[str, str]:
        prompt_inputs = dict(prompt_inputs)
        if "#context#" in parser.variable_keys:
            if context:
                prompt_inputs["#context#"] = context
            else:
                prompt_inputs["#context#"] = ""

        return prompt_inputs

    def _set_query_variable(
        self, query: str, parser: PromptTemplateParser, prompt_inputs: Mapping[str, str]
    ) -> Mapping[str, str]:
        prompt_inputs = dict(prompt_inputs)
        if "#query#" in parser.variable_keys:
            if query:
                prompt_inputs["#query#"] = query
            else:
                prompt_inputs["#query#"] = ""

        return prompt_inputs

    def _set_histories_variable(
        self,
        memory: TokenBufferMemory,
        memory_config: MemoryConfig,
        raw_prompt: str,
        role_prefix: MemoryConfig.RolePrefix,
        parser: PromptTemplateParser,
        prompt_inputs: Mapping[str, str],
        model_config: ModelConfigWithCredentialsEntity,
    ) -> Mapping[str, str]:
        prompt_inputs = dict(prompt_inputs)
        if "#histories#" in parser.variable_keys:
            if memory:
                inputs = {"#histories#": "", **prompt_inputs}
                parser = PromptTemplateParser(template=raw_prompt, with_variable_tmpl=self.with_variable_tmpl)
                prompt_inputs = {k: inputs[k] for k in parser.variable_keys if k in inputs}
                tmp_human_message = UserPromptMessage(content=parser.format(prompt_inputs))

                rest_tokens = self._calculate_rest_token([tmp_human_message], model_config)

                histories = self._get_history_messages_from_memory(
                    memory=memory,
                    memory_config=memory_config,
                    max_token_limit=rest_tokens,
                    human_prefix=role_prefix.user,
                    ai_prefix=role_prefix.assistant,
                )
                prompt_inputs["#histories#"] = histories
            else:
                prompt_inputs["#histories#"] = ""

        return prompt_inputs
