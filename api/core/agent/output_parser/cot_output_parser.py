import json
import re
from collections.abc import Generator
from typing import Union

from core.agent.entities import AgentScratchpadUnit
from core.model_runtime.entities.llm_entities import LLMResultChunk

# cdg:解析COT输出
class CotAgentOutputParser:
    @classmethodL
    def handle_react_stream_output(
            cls, llm_response: Generator[LLMResultChunk, None, None], usage_dict: dict
    ) -> Generator[Union[str, AgentScratchpadUnit.Action], None, None]:
        def parse_action(json_str):
            """
            解析给定的JSON字符串，提取动作名称和输入。

            参数:
            json_str (str): 包含动作信息的JSON字符串。

            返回:
            AgentScratchpadUnit.Action 或 str: 如果成功解析动作名称和输入，则返回相应的AgentScratchpadUnit.Action对象；
            否则返回原始的JSON字符串。
            """
            try:
                # cdg:尝试解析JSON字符串，允许非严格的JSON格式
                action = json.loads(json_str, strict=False)
                action_name = None
                action_input = None

                # cohere always returns a list
                if isinstance(action, list) and len(action) == 1:
                    action = action[0]

                # cdg:遍历动作字典，寻找动作名称和动作输入
                for key, value in action.items():
                    if "input" in key.lower():
                        action_input = value
                    else:
                        action_name = value

                # cdg:如果是Action，则将Action名称和Action的输入构建AgentScratchpadUnit输出
                if action_name is not None and action_input is not None:
                    return AgentScratchpadUnit.Action(
                        action_name=action_name,
                        action_input=action_input,
                    )
                else:
                    return json_str or ""
            except:
                return json_str or ""

        def extra_json_from_code_block(code_block) -> Generator[Union[str, AgentScratchpadUnit.Action], None, None]:
            code_blocks = re.findall(r"```(.*?)```", code_block, re.DOTALL)
            if not code_blocks:
                return
            for block in code_blocks:
                json_text = re.sub(r"^[a-zA-Z]+\n", "", block.strip(), flags=re.MULTILINE)
                yield parse_action(json_text)

        code_block_cache = ""
        code_block_delimiter_count = 0
        in_code_block = False
        json_cache = ""
        json_quote_count = 0
        in_json = False
        got_json = False

        action_cache = ""
        action_str = "action:"
        action_idx = 0

        thought_cache = ""
        thought_str = "thought:"
        thought_idx = 0

        last_character = ""

        for response in llm_response:
            if response.delta.usage:
                usage_dict["usage"] = response.delta.usage
            response_content = response.delta.message.content
            # cdg:正常情况下response.delta.message.content的格式为str类型
            if not isinstance(response_content, str):
                continue

            # stream
            index = 0
            while index < len(response_content):
                steps = 1
                delta = response_content[index: index + steps]
                yield_delta = False

                # cdg:判断是否是代码块的起始或结束，以"`"作为起始或结束的标识
                if delta == "`":
                    last_character = delta
                    code_block_cache += delta
                    code_block_delimiter_count += 1
                else:
                    # cdg:判断是否在代码块内的内容
                    if not in_code_block:
                        # cdg:如果不在代码块内，而且代码块的内容不为空，则将代码块的内容返回
                        if code_block_delimiter_count > 0:
                            last_character = delta
                            yield code_block_cache
                        code_block_cache = ""
                    else:
                        # cdg:如果在代码块内，则将代码块内容添加到缓存中
                        last_character = delta
                        code_block_cache += delta
                    code_block_delimiter_count = 0

                if not in_code_block and not in_json:
                    # cdg:action开始字符串处理
                    if delta.lower() == action_str[action_idx] and action_idx == 0:
                        if last_character not in {"\n", " ", ""}:
                            yield_delta = True
                        else:
                            last_character = delta
                            action_cache += delta
                            action_idx += 1
                            if action_idx == len(action_str):
                                action_cache = ""
                                action_idx = 0
                            index += steps
                            continue
                    elif delta.lower() == action_str[action_idx] and action_idx > 0:  # cdg:判断是否是action字符串
                        last_character = delta
                        action_cache += delta
                        action_idx += 1
                        if action_idx == len(action_str):
                            action_cache = ""
                            action_idx = 0
                        index += steps
                        continue
                    else:
                        if action_cache:
                            last_character = delta
                            yield action_cache
                            action_cache = ""
                            action_idx = 0

                    if delta.lower() == thought_str[thought_idx] and thought_idx == 0:
                        if last_character not in {"\n", " ", ""}:
                            yield_delta = True
                        else:
                            last_character = delta
                            thought_cache += delta
                            thought_idx += 1
                            if thought_idx == len(thought_str):
                                thought_cache = ""
                                thought_idx = 0
                            index += steps
                            continue
                    elif delta.lower() == thought_str[thought_idx] and thought_idx > 0:
                        last_character = delta
                        thought_cache += delta
                        thought_idx += 1
                        if thought_idx == len(thought_str):
                            thought_cache = ""
                            thought_idx = 0
                        index += steps
                        continue
                    else:
                        if thought_cache:
                            last_character = delta
                            yield thought_cache
                            thought_cache = ""
                            thought_idx = 0

                    if yield_delta:
                        index += steps
                        last_character = delta
                        yield delta
                        continue

                if code_block_delimiter_count == 3:
                    if in_code_block:
                        last_character = delta
                        yield from extra_json_from_code_block(code_block_cache)
                        code_block_cache = ""

                    in_code_block = not in_code_block
                    code_block_delimiter_count = 0

                if not in_code_block:
                    # handle single json
                    if delta == "{":
                        json_quote_count += 1
                        in_json = True
                        last_character = delta
                        json_cache += delta
                    elif delta == "}":
                        last_character = delta
                        json_cache += delta
                        if json_quote_count > 0:
                            json_quote_count -= 1
                            if json_quote_count == 0:
                                in_json = False
                                got_json = True
                                index += steps
                                continue
                    else:
                        if in_json:
                            last_character = delta
                            json_cache += delta

                    if got_json:
                        got_json = False
                        last_character = delta
                        yield parse_action(json_cache)
                        json_cache = ""
                        json_quote_count = 0
                        in_json = False

                if not in_code_block and not in_json:
                    last_character = delta
                    yield delta.replace("`", "")

                index += steps

        if code_block_cache:
            yield code_block_cache

        if json_cache:
            yield parse_action(json_cache)
