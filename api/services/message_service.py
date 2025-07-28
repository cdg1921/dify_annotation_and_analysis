import json
from typing import Optional, Union

from core.app.apps.advanced_chat.app_config_manager import AdvancedChatAppConfigManager
from core.app.entities.app_invoke_entities import InvokeFrom
from core.llm_generator.llm_generator import LLMGenerator
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.ops.entities.trace_entity import TraceTaskName
from core.ops.ops_trace_manager import TraceQueueManager, TraceTask
from core.ops.utils import measure_time
from extensions.ext_database import db
from libs.infinite_scroll_pagination import InfiniteScrollPagination
from models.account import Account
from models.model import App, AppMode, AppModelConfig, EndUser, Message, MessageFeedback
from services.conversation_service import ConversationService
from services.errors.conversation import ConversationCompletedError, ConversationNotExistsError
from services.errors.message import (
    FirstMessageNotExistsError,
    LastMessageNotExistsError,
    MessageNotExistsError,
    SuggestedQuestionsAfterAnswerDisabledError,
)
from services.workflow_service import WorkflowService

# cdg:消息服务，用于处理消息的创建、获取、分页、反馈等操作。
class MessageService:
    # cdg:分页获取消息，根据first_id获取消息，主要实现思路：
    # 1. 检查用户是否存在，如果用户不存在，则返回空结果。
    # 2. 检查会话ID是否存在，如果会话ID不存在，则返回空结果。
    # 3. 获取会话，如果会话不存在，则返回空结果。
    # 4. 如果first_id存在，则根据first_id获取消息，如果消息不存在，则抛出FirstMessageNotExistsError异常。
    # 5. 如果first_id不存在，则根据会话ID获取消息。
    # 6. 如果历史消息长度等于limit，则获取下一页的消息，同时判断是否还有更多消息。
    # 7. 如果历史消息长度小于limit，则返回空结果。
    # 8. 如果order为asc，则将历史消息反转。
    # 9. 返回分页结果。
    @classmethod
    def pagination_by_first_id(
        cls,
        app_model: App,
        user: Optional[Union[Account, EndUser]],
        conversation_id: str,
        first_id: Optional[str],
        limit: int,
        order: str = "asc",
    ) -> InfiniteScrollPagination:
        if not user:
            return InfiniteScrollPagination(data=[], limit=limit, has_more=False)

        if not conversation_id:
            return InfiniteScrollPagination(data=[], limit=limit, has_more=False)

        conversation = ConversationService.get_conversation(
            app_model=app_model, user=user, conversation_id=conversation_id
        )

        if first_id:
            first_message = (
                db.session.query(Message)
                .filter(Message.conversation_id == conversation.id, Message.id == first_id)
                .first()
            )

            if not first_message:
                raise FirstMessageNotExistsError()

            history_messages = (
                db.session.query(Message)
                .filter(
                    Message.conversation_id == conversation.id,
                    Message.created_at < first_message.created_at,
                    Message.id != first_message.id,
                )
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )
        else:
            history_messages = (
                db.session.query(Message)
                .filter(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )

        has_more = False
        if len(history_messages) == limit:
            current_page_first_message = history_messages[-1]
            rest_count = (
                db.session.query(Message)
                .filter(
                    Message.conversation_id == conversation.id,
                    Message.created_at < current_page_first_message.created_at,
                    Message.id != current_page_first_message.id,
                )
                .count()
            )

            if rest_count > 0:
                has_more = True

        if order == "asc":
            history_messages = list(reversed(history_messages))

        return InfiniteScrollPagination(data=history_messages, limit=limit, has_more=has_more)

    # cdg:分页获取消息，根据last_id获取消息，主要实现思路：
    # 1. 检查用户是否存在，如果用户不存在，则返回空结果。
    # 2. 检查会话ID是否存在，如果会话ID不存在，则返回空结果。
    # 3. 获取会话，如果会话不存在，则返回空结果。
    # 4. 如果last_id存在，则根据last_id获取消息，如果消息不存在，则抛出LastMessageNotExistsError异常。
    # 5. 如果last_id不存在，则根据会话ID获取消息。
    # 6. 如果历史消息长度等于limit，则获取下一页的消息，同时判断是否还有更多消息。
    @classmethod
    def pagination_by_last_id(
        cls,
        app_model: App,
        user: Optional[Union[Account, EndUser]],
        last_id: Optional[str],
        limit: int,
        conversation_id: Optional[str] = None,
        include_ids: Optional[list] = None,
    ) -> InfiniteScrollPagination:
        if not user:
            return InfiniteScrollPagination(data=[], limit=limit, has_more=False)

        base_query = db.session.query(Message)

        if conversation_id is not None:
            conversation = ConversationService.get_conversation(
                app_model=app_model, user=user, conversation_id=conversation_id
            )

            base_query = base_query.filter(Message.conversation_id == conversation.id)

        if include_ids is not None:
            base_query = base_query.filter(Message.id.in_(include_ids))

        if last_id:
            last_message = base_query.filter(Message.id == last_id).first()

            if not last_message:
                raise LastMessageNotExistsError()

            history_messages = (
                base_query.filter(Message.created_at < last_message.created_at, Message.id != last_message.id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )
        else:
            history_messages = base_query.order_by(Message.created_at.desc()).limit(limit).all()

        has_more = False
        if len(history_messages) == limit:
            current_page_first_message = history_messages[-1]
            rest_count = base_query.filter(
                Message.created_at < current_page_first_message.created_at, Message.id != current_page_first_message.id
            ).count()

            if rest_count > 0:
                has_more = True

        return InfiniteScrollPagination(data=history_messages, limit=limit, has_more=has_more)

    # cdg:创建反馈消息
    @classmethod
    def create_feedback(
        cls,
        *,
        app_model: App,
        message_id: str,
        user: Optional[Union[Account, EndUser]],
        rating: Optional[str],
        content: Optional[str],
    ):
        if not user:
            raise ValueError("user cannot be None")

        message = cls.get_message(app_model=app_model, user=user, message_id=message_id)

        feedback = message.user_feedback if isinstance(user, EndUser) else message.admin_feedback

        # cdg:如果rating为空而且反馈存在，则删除反馈。
        if not rating and feedback:
            db.session.delete(feedback)
        # cdg:如果rating不为空而且反馈存在，则更新反馈。
        elif rating and feedback:
            feedback.rating = rating
            feedback.content = content
        # cdg:如果rating为空而且反馈不存在，则抛出ValueError异常。
        elif not rating and not feedback:
            raise ValueError("rating cannot be None when feedback not exists")
        # cdg:如果rating不为空而且反馈不存在，则创建反馈。
        else:
            feedback = MessageFeedback(
                app_id=app_model.id,
                conversation_id=message.conversation_id,
                message_id=message.id,
                rating=rating,
                content=content,
                from_source=("user" if isinstance(user, EndUser) else "admin"),
                from_end_user_id=(user.id if isinstance(user, EndUser) else None),
                from_account_id=(user.id if isinstance(user, Account) else None),
            )
            db.session.add(feedback)

        db.session.commit()

        return feedback

    # cdg:获取消息
    @classmethod
    def get_message(cls, app_model: App, user: Optional[Union[Account, EndUser]], message_id: str):
        message = (
            db.session.query(Message)
            .filter(
                Message.id == message_id,
                Message.app_id == app_model.id,
                Message.from_source == ("api" if isinstance(user, EndUser) else "console"),
                Message.from_end_user_id == (user.id if isinstance(user, EndUser) else None),
                Message.from_account_id == (user.id if isinstance(user, Account) else None),
            )
            .first()
        )

        if not message:
            raise MessageNotExistsError()

        return message

    # cdg:获取建议问题
    @classmethod
    def get_suggested_questions_after_answer(
        cls, app_model: App, user: Optional[Union[Account, EndUser]], message_id: str, invoke_from: InvokeFrom
    ) -> list[Message]:
        if not user:
            raise ValueError("user cannot be None")
        # cdg:获取历史消息
        message = cls.get_message(app_model=app_model, user=user, message_id=message_id)
        # cdg:获取会话
        conversation = ConversationService.get_conversation(
            app_model=app_model, conversation_id=message.conversation_id, user=user
        )
        # cdg:如果会话不存在，则抛出ConversationNotExistsError异常。
        if not conversation:
            raise ConversationNotExistsError()
        # cdg:如果会话状态不为normal，则抛出ConversationCompletedError异常。
        if conversation.status != "normal":
            raise ConversationCompletedError()
        # cdg:获取模型管理器
        model_manager = ModelManager()
        # cdg:如果应用模式为高级聊天，则获取工作流服务。
        if app_model.mode == AppMode.ADVANCED_CHAT.value:
            workflow_service = WorkflowService()
            if invoke_from == InvokeFrom.DEBUGGER:
                workflow = workflow_service.get_draft_workflow(app_model=app_model)
            else:
                workflow = workflow_service.get_published_workflow(app_model=app_model)

            if workflow is None:
                return []
            # cdg:获取应用配置，目的是获取应用配置中的额外功能中的建议问题功能是否开启。
            app_config = AdvancedChatAppConfigManager.get_app_config(app_model=app_model, workflow=workflow)

            # cdg:如果应用配置的额外功能中的建议问题为空，则抛出SuggestedQuestionsAfterAnswerDisabledError异常。
            if not app_config.additional_features.suggested_questions_after_answer:
                raise SuggestedQuestionsAfterAnswerDisabledError()
            # cdg:获取默认模型实例
            model_instance = model_manager.get_default_model_instance(
                tenant_id=app_model.tenant_id, model_type=ModelType.LLM
            )
        else:
            # cdg:如果会话的模型配置为空，则获取应用配置。
            if not conversation.override_model_configs:
                app_model_config = (
                    db.session.query(AppModelConfig)
                    .filter(
                        AppModelConfig.id == conversation.app_model_config_id, AppModelConfig.app_id == app_model.id
                    )
                    .first()
                )
            # cdg:如果会话的模型配置不为空，则获取会话的模型配置。
            else:
                # cdg:将会话的模型配置转换为字典。
                conversation_override_model_configs = json.loads(conversation.override_model_configs)
                app_model_config = AppModelConfig(
                    id=conversation.app_model_config_id,
                    app_id=app_model.id,
                )
                # cdg:将会话的模型配置转换为应用配置。
                app_model_config = app_model_config.from_model_config_dict(conversation_override_model_configs)
            # cdg:如果应用配置为空，则抛出ValueError异常。
            if not app_model_config:
                raise ValueError("did not find app model config")
            # cdg:获取应用配置中的建议问题功能是否开启，如果为False，则抛出SuggestedQuestionsAfterAnswerDisabledError异常。
            suggested_questions_after_answer = app_model_config.suggested_questions_after_answer_dict
            if suggested_questions_after_answer.get("enabled", False) is False:
                raise SuggestedQuestionsAfterAnswerDisabledError()
            # cdg:获取模型实例
            model_instance = model_manager.get_model_instance(
                tenant_id=app_model.tenant_id,
                provider=app_model_config.model_dict["provider"],
                model_type=ModelType.LLM,
                model=app_model_config.model_dict["name"],
            )

        # cdg:实例化TokenBufferMemory，用于获取会话的历史消息。
        memory = TokenBufferMemory(conversation=conversation, model_instance=model_instance)
        # cdg:获取会话的历史消息
        histories = memory.get_history_prompt_text(
            max_token_limit=3000,
            message_limit=3,
        )
        # cdg:根据历史消息生成建议问题
        with measure_time() as timer:
            questions: list[Message] = LLMGenerator.generate_suggested_questions_after_answer(
                tenant_id=app_model.tenant_id, histories=histories
            )

        # cdg:添加追踪任务，对底层追踪任务的封装，简化了追踪任务的添加过程。
        # get tracing instance
        trace_manager = TraceQueueManager(app_id=app_model.id)
        trace_manager.add_trace_task(
            TraceTask(
                TraceTaskName.SUGGESTED_QUESTION_TRACE, message_id=message_id, suggested_question=questions, timer=timer
            )
        )

        return questions
