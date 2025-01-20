import json
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from typing import Any, Optional

from core.agent.base_agent_runner import BaseAgentRunner
from core.agent.entities import AgentScratchpadUnit
from core.agent.output_parser.cot_output_parser import CotAgentOutputParser
from core.app.apps.base_app_queue_manager import PublishFrom
from core.app.entities.queue_entities import QueueAgentThoughtEvent, QueueMessageEndEvent, QueueMessageFileEvent
from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.ops.ops_trace_manager import TraceQueueManager
from core.prompt.agent_history_prompt_transform import AgentHistoryPromptTransform
from core.tools.entities.tool_entities import ToolInvokeMeta
from core.tools.tool.tool import Tool
from core.tools.tool_engine import ToolEngine
from models.model import Message

# cdg:CotAgentRunner -> BaseAgentRunner -> AppRunner
class CotAgentRunner(BaseAgentRunner, ABC):
    _is_first_iteration = True                                     # cdg:是否首次迭代，默认为True
    _ignore_observation_providers = ["wenxin"]                     # cdg:忽略文心一言模型
    _historic_prompt_messages: list[PromptMessage] | None = None   # cdg:历史会话记录
    # cdg:Agent Scratchpad（代理草稿板）是一个用于增强智能代理（如聊天机器人或自动化系统）功能的工具。
    # Agent Scratchpad提供一个临时的存储空间，用于保存代理在处理任务时生成的中间结果、思路或数据,
    # 使得代理能够在执行复杂任务时保持上下文。
    _agent_scratchpad: list[AgentScratchpadUnit] | None = None
    _instruction: str = ""  # FIXME this must be str for now       # cdg:开场白
    _query: str | None = None
    _prompt_messages_tools: list[PromptMessageTool] = []

    def run(
        self,
        message: Message,
        query: str,
        inputs: Mapping[str, str],
    ) -> Generator:
        """
        Run Cot agent application
        """
        app_generate_entity = self.application_generate_entity
        self._repack_app_generate_entity(app_generate_entity)  # cdg:生成式实体（大模型）包装，实际上就是当提示词模板为空时，设置为空值
        # cdg:初始化COT状态
        self._init_react_state(query)

        # cdg:任务跟踪器
        trace_manager = app_generate_entity.trace_manager

        # check model mode
        if "Observation" not in app_generate_entity.model_conf.stop:
            if app_generate_entity.model_conf.provider not in self._ignore_observation_providers:
                app_generate_entity.model_conf.stop.append("Observation")

        app_config = self.app_config

        # init instruction
        inputs = inputs or {}
        instruction = app_config.prompt_template.simple_prompt_template
        self._instruction = self._fill_in_inputs_from_external_data_tools(instruction=instruction or "", inputs=inputs)

        iteration_step = 1
        # cdg:max_iteration_steps不小于6
        max_iteration_steps = min(app_config.agent.max_iteration if app_config.agent else 5, 5) + 1

        # convert tools into ModelRuntime Tool format
        tool_instances, self._prompt_messages_tools = self._init_prompt_tools()

        function_call_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        # cdg:计算已使用的token数量
        def increase_usage(final_llm_usage_dict: dict[str, Optional[LLMUsage]], usage: LLMUsage):
            if not final_llm_usage_dict["usage"]:
                final_llm_usage_dict["usage"] = usage
            else:
                llm_usage = final_llm_usage_dict["usage"]
                llm_usage.prompt_tokens += usage.prompt_tokens
                llm_usage.completion_tokens += usage.completion_tokens
                llm_usage.prompt_price += usage.prompt_price
                llm_usage.completion_price += usage.completion_price
                llm_usage.total_price += usage.total_price

        model_instance = self.model_instance

        while function_call_state and iteration_step <= max_iteration_steps:
            # continue to run until there is not any tool call
            # cdg:在给定的steps内循环执行直至没有工具可以调用
            function_call_state = False

            # cdg:最后一步进行总结，不再调用工具，否则就没有总结的步骤，直接输出工具调用的结果是不合适的
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # cdg:创建包含智能体思考内容的消息对象，并更新到数据库中
            agent_thought = self.create_agent_thought(
                message_id=message.id, message="", tool_name="", tool_input="", messages_ids=message_file_ids
            )

            # cdg:Agent执行过程记录到消息队列中，中间过程为思考过程，Event类型为QueueAgentThoughtEvent
            if iteration_step > 1:
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought.id), PublishFrom.APPLICATION_MANAGER
                )

            # recalc llm max tokens
            # cdg:更新提示词消息并重新计算可用的max_tokens
            prompt_messages = self._organize_prompt_messages()
            self.recalc_llm_max_tokens(self.model_config, prompt_messages)
            # invoke model
            # cdg:调用大模型进行思考，并返回大模型生成的结果
            chunks = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters=app_generate_entity.model_conf.parameters,
                tools=[],
                stop=app_generate_entity.model_conf.stop,
                stream=True,
                user=self.user_id,
                callbacks=[],
            )

            # cdg:强制为流式输出，为什么不能是blocking输出？
            if not isinstance(chunks, Generator):
                raise ValueError("Expected streaming response from LLM")

            # cdg:通过检查返回结果是否为空，以判断大模型调用是否成功
            # check llm result
            if not chunks:
                raise ValueError("failed to invoke llm")

            # cdg:tokens使用情况初始化
            usage_dict: dict[str, Optional[LLMUsage]] = {"usage": None}

            # cdg:大模型返回结果整理，返回结果为Markdown格式信息，可能包含多种不同格式内容，如代码、json等，
            react_chunks = CotAgentOutputParser.handle_react_stream_output(chunks, usage_dict)

            # cdg:初始化一个AgentScratchpadUnit对象
            scratchpad = AgentScratchpadUnit(
                agent_response="",
                thought="",
                action_str="",
                observation="",
                action=None,
            )

            # publish agent thought if it's first iteration
            if iteration_step == 1:
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought.id), PublishFrom.APPLICATION_MANAGER
                )

            for chunk in react_chunks:
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    action = chunk
                    # detect action
                    if scratchpad.agent_response is not None:
                        scratchpad.agent_response += json.dumps(chunk.model_dump())
                    scratchpad.action_str = json.dumps(chunk.model_dump())
                    scratchpad.action = action
                else:
                    if scratchpad.agent_response is not None:
                        scratchpad.agent_response += chunk
                    if scratchpad.thought is not None:
                        scratchpad.thought += chunk
                    yield LLMResultChunk(
                        model=self.model_config.model,
                        prompt_messages=prompt_messages,
                        system_fingerprint="",
                        delta=LLMResultChunkDelta(index=0, message=AssistantPromptMessage(content=chunk), usage=None),
                    )
            if scratchpad.thought is not None:
                scratchpad.thought = scratchpad.thought.strip() or "I am thinking about how to help you"
            if self._agent_scratchpad is not None:
                self._agent_scratchpad.append(scratchpad)

            # get llm usage
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            self.save_agent_thought(
                agent_thought=agent_thought,
                tool_name=scratchpad.action.action_name if scratchpad.action else "",
                tool_input={scratchpad.action.action_name: scratchpad.action.action_input} if scratchpad.action else {},
                tool_invoke_meta={},
                thought=scratchpad.thought or "",
                observation="",
                answer=scratchpad.agent_response or "",
                messages_ids=[],
                llm_usage=usage_dict["usage"],
            )

            if not scratchpad.is_final():
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought.id), PublishFrom.APPLICATION_MANAGER
                )

            if not scratchpad.action:
                # failed to extract action, return final answer directly
                final_answer = ""
            else:
                if scratchpad.action.action_name.lower() == "final answer":
                    # action is final answer, return final answer directly
                    try:
                        if isinstance(scratchpad.action.action_input, dict):
                            final_answer = json.dumps(scratchpad.action.action_input)
                        elif isinstance(scratchpad.action.action_input, str):
                            final_answer = scratchpad.action.action_input
                        else:
                            final_answer = f"{scratchpad.action.action_input}"
                    except json.JSONDecodeError:
                        final_answer = f"{scratchpad.action.action_input}"
                else:
                    function_call_state = True
                    # action is tool call, invoke tool
                    tool_invoke_response, tool_invoke_meta = self._handle_invoke_action(
                        action=scratchpad.action,
                        tool_instances=tool_instances,
                        message_file_ids=message_file_ids,
                        trace_manager=trace_manager,
                    )
                    scratchpad.observation = tool_invoke_response
                    scratchpad.agent_response = tool_invoke_response

                    self.save_agent_thought(
                        agent_thought=agent_thought,
                        tool_name=scratchpad.action.action_name,
                        tool_input={scratchpad.action.action_name: scratchpad.action.action_input},
                        thought=scratchpad.thought or "",
                        observation={scratchpad.action.action_name: tool_invoke_response},
                        tool_invoke_meta={scratchpad.action.action_name: tool_invoke_meta.to_dict()},
                        answer=scratchpad.agent_response,
                        messages_ids=message_file_ids,
                        llm_usage=usage_dict["usage"],
                    )

                    self.queue_manager.publish(
                        QueueAgentThoughtEvent(agent_thought_id=agent_thought.id), PublishFrom.APPLICATION_MANAGER
                    )

                # update prompt tool message
                for prompt_tool in self._prompt_messages_tools:
                    self.update_prompt_message_tool(tool_instances[prompt_tool.name], prompt_tool)

            iteration_step += 1

        yield LLMResultChunk(
            model=model_instance.model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0, message=AssistantPromptMessage(content=final_answer), usage=llm_usage["usage"]
            ),
            system_fingerprint="",
        )

        # save agent thought
        self.save_agent_thought(
            agent_thought=agent_thought,
            tool_name="",
            tool_input={},
            tool_invoke_meta={},
            thought=final_answer,
            observation={},
            answer=final_answer,
            messages_ids=[],
        )
        if self.variables_pool is not None and self.db_variables_pool is not None:
            self.update_db_variables(self.variables_pool, self.db_variables_pool)
        # publish end event
        self.queue_manager.publish(
            QueueMessageEndEvent(
                llm_result=LLMResult(
                    model=model_instance.model,
                    prompt_messages=prompt_messages,
                    message=AssistantPromptMessage(content=final_answer),
                    usage=llm_usage["usage"] or LLMUsage.empty_usage(),
                    system_fingerprint="",
                )
            ),
            PublishFrom.APPLICATION_MANAGER,
        )

    def _handle_invoke_action(
        self,
        action: AgentScratchpadUnit.Action,
        tool_instances: dict[str, Tool],
        message_file_ids: list[str],
        trace_manager: Optional[TraceQueueManager] = None,
    ) -> tuple[str, ToolInvokeMeta]:
        """
        handle invoke action
        :param action: action
        :param tool_instances: tool instances
        :param message_file_ids: message file ids
        :param trace_manager: trace manager
        :return: observation, meta
        """
        # action is tool call, invoke tool
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)

        if not tool_instance:
            answer = f"there is not a tool named {tool_call_name}"
            return answer, ToolInvokeMeta.error_instance(answer)

        if isinstance(tool_call_args, str):
            try:
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError:
                pass

        # invoke tool
        tool_invoke_response, message_files, tool_invoke_meta = ToolEngine.agent_invoke(
            tool=tool_instance,
            tool_parameters=tool_call_args,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            message=self.message,
            invoke_from=self.application_generate_entity.invoke_from,
            agent_tool_callback=self.agent_callback,
            trace_manager=trace_manager,
        )

        # publish files
        for message_file_id, save_as in message_files:
            if save_as is not None and self.variables_pool:
                # FIXME the save_as type is confusing, it should be a string or not
                self.variables_pool.set_file(tool_name=tool_call_name, value=message_file_id, name=str(save_as))

            # publish message file
            self.queue_manager.publish(
                QueueMessageFileEvent(message_file_id=message_file_id), PublishFrom.APPLICATION_MANAGER
            )
            # add message file ids
            message_file_ids.append(message_file_id)

        return tool_invoke_response, tool_invoke_meta

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        convert dict to action
        """
        return AgentScratchpadUnit.Action(action_name=action["action"], action_input=action["action_input"])

    def _fill_in_inputs_from_external_data_tools(self, instruction: str, inputs: Mapping[str, Any]) -> str:
        """
        fill in inputs from external data tools
        """
        for key, value in inputs.items():
            try:
                instruction = instruction.replace(f"{{{{{key}}}}}", str(value))
            except Exception as e:
                continue

        return instruction

    def _init_react_state(self, query) -> None:
        """
        init agent scratchpad
        """
        # cdg:初始化agent scratchpad。agent scratchpad与常见LLM scratchpad的区别
        self._query = query
        self._agent_scratchpad = []
        self._historic_prompt_messages = self._organize_historic_prompt_messages()

    @abstractmethod
    def _organize_prompt_messages(self) -> list[PromptMessage]:
        """
        organize prompt messages
        """

    def _format_assistant_message(self, agent_scratchpad: list[AgentScratchpadUnit]) -> str:
        """
        format assistant message
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message

    def _organize_historic_prompt_messages(
        self, current_session_messages: Optional[list[PromptMessage]] = None
    ) -> list[PromptMessage]:
        """
        organize historic prompt messages
        """
        result: list[PromptMessage] = []
        scratchpads: list[AgentScratchpadUnit] = []
        current_scratchpad: AgentScratchpadUnit | None = None

        for message in self.history_prompt_messages:
            if isinstance(message, AssistantPromptMessage):
                if not current_scratchpad:
                    if not isinstance(message.content, str | None):
                        raise NotImplementedError("expected str type")
                    current_scratchpad = AgentScratchpadUnit(
                        agent_response=message.content,
                        thought=message.content or "I am thinking about how to help you",
                        action_str="",
                        action=None,
                        observation=None,
                    )
                    scratchpads.append(current_scratchpad)
                if message.tool_calls:
                    try:
                        current_scratchpad.action = AgentScratchpadUnit.Action(
                            action_name=message.tool_calls[0].function.name,
                            action_input=json.loads(message.tool_calls[0].function.arguments),
                        )
                        current_scratchpad.action_str = json.dumps(current_scratchpad.action.to_dict())
                    except:
                        pass
            elif isinstance(message, ToolPromptMessage):
                if not current_scratchpad:
                    continue
                if isinstance(message.content, str):
                    current_scratchpad.observation = message.content
                else:
                    raise NotImplementedError("expected str type")
            elif isinstance(message, UserPromptMessage):
                if scratchpads:
                    result.append(AssistantPromptMessage(content=self._format_assistant_message(scratchpads)))
                    scratchpads = []
                    current_scratchpad = None

                result.append(message)

        if scratchpads:
            result.append(AssistantPromptMessage(content=self._format_assistant_message(scratchpads)))

        historic_prompts = AgentHistoryPromptTransform(
            model_config=self.model_config,
            prompt_messages=current_session_messages or [],
            history_messages=result,
            memory=self.memory,
        ).get_prompt()
        return historic_prompts
