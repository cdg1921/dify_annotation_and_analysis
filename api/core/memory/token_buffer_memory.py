from collections.abc import Sequence
from typing import Optional

from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.file import file_manager
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContent,
    PromptMessageRole,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.prompt.utils.extract_thread_messages import extract_thread_messages
from extensions.ext_database import db
from factories import file_factory
from models.model import AppMode, Conversation, Message, MessageFile
from models.workflow import WorkflowRun

# cdg:TokenBufferMemory，短期记忆缓冲器，实现了获取会话的历史消息和根据历史消息生成提示消息的功能。
# cdg:每个会话的TokenBufferMemory实例是独立的，每个实例都对应一个会话。
class TokenBufferMemory:
    def __init__(self, conversation: Conversation, model_instance: ModelInstance) -> None:
        self.conversation = conversation
        self.model_instance = model_instance # cdg:模型实例，用于计算历史消息的token数量。

    # cdg:获取会话的历史消息
    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: Optional[int] = None
    ) -> Sequence[PromptMessage]:
        """
        Get history prompt messages.
        :param max_token_limit: max token limit
        :param message_limit: message limit
        """
        app_record = self.conversation.app

        # fetch limited messages, and return reversed
        query = (
            db.session.query(
                Message.id,
                Message.query,
                Message.answer,
                Message.created_at,
                Message.workflow_run_id,
                Message.parent_message_id,
            )
            .filter(
                Message.conversation_id == self.conversation.id,
            )
            .order_by(Message.created_at.desc())  # cdg:按照创建时间降序排序，最新的消息排在最前面。
        )
        # cdg:如果message_limit不为空且大于0，则将message_limit限制在500以内。
        if message_limit and message_limit > 0:
            message_limit = min(message_limit, 500)
        # cdg:如果message_limit为空或小于等于0，则将message_limit设置为500。
        else:
            message_limit = 500
        # cdg:获取消息，最多获取message_limit条消息。
        messages = query.limit(message_limit).all()

        # cdg:获取消息的线程消息，线程消息是消息的子消息。与获取全部消息相比，线程消息只包含与最新消息相关的消息。
        # instead of all messages from the conversation, we only need to extract messages
        # that belong to the thread of last message
        thread_messages = extract_thread_messages(messages)

        # cdg:如果线程消息不为空，且线程消息的第一个消息的答案为空，则将线程消息的第一个消息删除。
        # for newly created message, its answer is temporarily empty, we don't need to add it to memory
        if thread_messages and not thread_messages[0].answer:
            thread_messages.pop(0)

        # cdg:将线程消息反转，最新的消息排在最前面。
        messages = list(reversed(thread_messages))

        # cdg:创建提示消息列表。根据有无文件情况构建提示消息。
        prompt_messages: list[PromptMessage] = []
        for message in messages:
            # cdg:获取消息的文件。
            files = db.session.query(MessageFile).filter(MessageFile.message_id == message.id).all()
            # cdg:如果消息有文件，则将文件添加到提示消息列表中。
            if files:
                file_extra_config = None
                if self.conversation.mode not in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
                    file_extra_config = FileUploadConfigManager.convert(self.conversation.model_config)
                else:
                    if message.workflow_run_id:
                        workflow_run = (
                            db.session.query(WorkflowRun).filter(WorkflowRun.id == message.workflow_run_id).first()
                        )

                        if workflow_run and workflow_run.workflow:
                            file_extra_config = FileUploadConfigManager.convert(
                                workflow_run.workflow.features_dict, is_vision=False
                            )

                detail = ImagePromptMessageContent.DETAIL.LOW
                if file_extra_config and app_record:
                    file_objs = file_factory.build_from_message_files(
                        message_files=files, tenant_id=app_record.tenant_id, config=file_extra_config
                    )
                    if file_extra_config.image_config and file_extra_config.image_config.detail:
                        detail = file_extra_config.image_config.detail
                else:
                    file_objs = []
                # cdg:如果消息没有文件，则将消息添加到提示消息列表中。
                if not file_objs:
                    prompt_messages.append(UserPromptMessage(content=message.query))
                else:
                    prompt_message_contents: list[PromptMessageContent] = []
                    prompt_message_contents.append(TextPromptMessageContent(data=message.query))
                    for file in file_objs:
                        prompt_message = file_manager.to_prompt_message_content(
                            file,
                            image_detail_config=detail,
                        )
                        prompt_message_contents.append(prompt_message)

                    prompt_messages.append(UserPromptMessage(content=prompt_message_contents))

            else:
                prompt_messages.append(UserPromptMessage(content=message.query))

            prompt_messages.append(AssistantPromptMessage(content=message.answer))

        if not prompt_messages:
            return []

        # cdg:计算提示消息的token数量。
        # prune the chat message if it exceeds the max token limit
        curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)

        # cdg:如果提示消息的token数量大于最大token限制，则将提示消息修剪到最大token限制。
        if curr_message_tokens > max_token_limit:
            pruned_memory = []
            while curr_message_tokens > max_token_limit and len(prompt_messages) > 1:
                pruned_memory.append(prompt_messages.pop(0))
                curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)

        return prompt_messages

    # cdg:根据历史消息生成提示消息
    def get_history_prompt_text(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "Assistant",
        max_token_limit: int = 2000,
        message_limit: Optional[int] = None,
    ) -> str:
        """
        Get history prompt text.
        :param human_prefix: human prefix
        :param ai_prefix: ai prefix
        :param max_token_limit: max token limit
        :param message_limit: message limit
        :return:
        """
        # cdg:根据指定长度将历史消息转为文本
        prompt_messages = self.get_history_prompt_messages(max_token_limit=max_token_limit, message_limit=message_limit)

        string_messages = []
        for m in prompt_messages:
            # cdg:仅将UserPromptMessage和AssistantPromptMessage添加到历史消息中，忽略SystemPromptMessage等消息
            if m.role == PromptMessageRole.USER:
                role = human_prefix
            elif m.role == PromptMessageRole.ASSISTANT:
                role = ai_prefix
            else:
                continue

            if isinstance(m.content, list):
                inner_msg = ""
                for content in m.content:
                    if isinstance(content, TextPromptMessageContent):
                        inner_msg += f"{content.data}\n"
                    elif isinstance(content, ImagePromptMessageContent):
                        inner_msg += "[image]\n"

                string_messages.append(f"{role}: {inner_msg.strip()}")
            else:
                message = f"{role}: {m.content}"
                string_messages.append(message)

        # cdg:将提示消息列表拼接成字符串，每个提示消息之间用换行符分隔。
        return "\n".join(string_messages)
