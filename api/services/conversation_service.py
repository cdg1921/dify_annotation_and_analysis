from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Optional, Union

from sqlalchemy import asc, desc, func, or_, select
from sqlalchemy.orm import Session

from core.app.entities.app_invoke_entities import InvokeFrom
from core.llm_generator.llm_generator import LLMGenerator
from extensions.ext_database import db
from libs.infinite_scroll_pagination import InfiniteScrollPagination
from models.account import Account
from models.model import App, Conversation, EndUser, Message
from services.errors.conversation import ConversationNotExistsError, LastConversationNotExistsError
from services.errors.message import MessageNotExistsError

# cdg: 会话服务，用于处理会话的创建、删除、重命名、自动生成名称、获取会话、删除会话。
class ConversationService:
    # cdg: 分页获取会话。具体实现思路如下：
    # 1.检查用户是否为空，如果为空，则返回空的分页结果。
    # 2.构建会话查询语句，包括过滤条件、排序条件、分页条件。
    # 3.如果last_id不为空，则获取上一页的会话，并构建过滤条件。
    # 4.执行查询语句，获取会话列表。
    # 5.检查是否还有更多数据，如果有，则返回True，否则返回False。
    # 6.返回分页结果。
    @classmethod
    def pagination_by_last_id(
        cls,
        *,
        session: Session,
        app_model: App,
        user: Optional[Union[Account, EndUser]],
        last_id: Optional[str],
        limit: int,
        invoke_from: InvokeFrom,
        include_ids: Optional[Sequence[str]] = None,
        exclude_ids: Optional[Sequence[str]] = None,
        sort_by: str = "-updated_at",
    ) -> InfiniteScrollPagination:
        # cdg: 如果用户为空，则返回空的分页结果。
        if not user:
            return InfiniteScrollPagination(data=[], limit=limit, has_more=False)

        # cdg: 构建SQLAlchemy会话查询语句，包括过滤条件、排序条件、分页条件。
        stmt = select(Conversation).where(
            Conversation.is_deleted == False,
            Conversation.app_id == app_model.id,
            Conversation.from_source == ("api" if isinstance(user, EndUser) else "console"),
            Conversation.from_end_user_id == (user.id if isinstance(user, EndUser) else None),
            Conversation.from_account_id == (user.id if isinstance(user, Account) else None),
            or_(Conversation.invoke_from.is_(None), Conversation.invoke_from == invoke_from.value),
        )
        # cdg: 如果include_ids不为空，则添加过滤条件。
        if include_ids is not None:
            stmt = stmt.where(Conversation.id.in_(include_ids))
        # cdg: 如果exclude_ids不为空，则添加过滤条件。
        if exclude_ids is not None:
            stmt = stmt.where(~Conversation.id.in_(exclude_ids))

        # cdg: 定义排序字段和方向。
        # define sort fields and directions
        sort_field, sort_direction = cls._get_sort_params(sort_by)

        # cdg: 如果last_id不为空，则获取上一页的会话，并构建过滤条件。
        if last_id:
            last_conversation = session.scalar(stmt.where(Conversation.id == last_id))
            if not last_conversation:
                raise LastConversationNotExistsError()

            # build filters based on sorting
            filter_condition = cls._build_filter_condition(
                sort_field=sort_field,
                sort_direction=sort_direction,
                reference_conversation=last_conversation,
            )
            stmt = stmt.where(filter_condition)
        query_stmt = stmt.order_by(sort_direction(getattr(Conversation, sort_field))).limit(limit)
        conversations = session.scalars(query_stmt).all()

        # cdg: 检查是否还有更多数据，如果有，则返回True，否则返回False。
        has_more = False
        if len(conversations) == limit:
            current_page_last_conversation = conversations[-1]
            rest_filter_condition = cls._build_filter_condition(
                sort_field=sort_field,
                sort_direction=sort_direction,
                reference_conversation=current_page_last_conversation,
            )
            count_stmt = select(func.count()).select_from(stmt.where(rest_filter_condition).subquery())
            rest_count = session.scalar(count_stmt) or 0
            if rest_count > 0:
                has_more = True

        return InfiniteScrollPagination(data=conversations, limit=limit, has_more=has_more)

    # cdg: 获取排序参数。
    @classmethod
    def _get_sort_params(cls, sort_by: str):
        if sort_by.startswith("-"):
            return sort_by[1:], desc
        return sort_by, asc

    # cdg: 构建过滤条件。
    @classmethod
    def _build_filter_condition(cls, sort_field: str, sort_direction: Callable, reference_conversation: Conversation):
        field_value = getattr(reference_conversation, sort_field)
        if sort_direction == desc:
            return getattr(Conversation, sort_field) < field_value
        else:
            return getattr(Conversation, sort_field) > field_value

    # cdg: 重命名会话。
    @classmethod
    def rename(
        cls,
        app_model: App,
        conversation_id: str,
        user: Optional[Union[Account, EndUser]],
        name: str,
        auto_generate: bool,
    ):
        conversation = cls.get_conversation(app_model, conversation_id, user)

        if auto_generate:
            return cls.auto_generate_name(app_model, conversation)
        else:
            conversation.name = name
            conversation.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.session.commit()

        return conversation

    # cdg: 自动生成会话名称。
    @classmethod
    def auto_generate_name(cls, app_model: App, conversation: Conversation):
        # get conversation first message
        message = (
            db.session.query(Message)
            .filter(Message.app_id == app_model.id, Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
            .first()
        )

        if not message:
            raise MessageNotExistsError()

        # generate conversation name
        try:
            # cdg: 调用LLMGenerator生成会话名称。
            name = LLMGenerator.generate_conversation_name(
                app_model.tenant_id, message.query, conversation.id, app_model.id
            )
            # cdg: 更新会话名称。
            conversation.name = name
        except:
            pass

        db.session.commit()

        return conversation

    # cdg: 获取会话。具体实现思路如下：
    # 1.构建会话查询语句，包括过滤条件、排序条件、分页条件。
    # 2.执行查询语句，获取会话列表。
    # 3.检查会话是否存在，如果不存在，则抛出ConversationNotExistsError异常。
    # 4.返回会话。
    @classmethod
    def get_conversation(cls, app_model: App, conversation_id: str, user: Optional[Union[Account, EndUser]]):
        conversation = (
            db.session.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.app_id == app_model.id,
                Conversation.from_source == ("api" if isinstance(user, EndUser) else "console"),
                Conversation.from_end_user_id == (user.id if isinstance(user, EndUser) else None),
                Conversation.from_account_id == (user.id if isinstance(user, Account) else None),
                Conversation.is_deleted == False,
            )
            .first()
        )

        if not conversation:
            raise ConversationNotExistsError()

        return conversation

    # cdg: 删除会话。具体实现思路如下：
    # 1.获取会话。
    # 2.设置会话为已删除。
    # 3.更新会话更新时间。
    # 4.提交删除操作。
    @classmethod
    def delete(cls, app_model: App, conversation_id: str, user: Optional[Union[Account, EndUser]]):
        conversation = cls.get_conversation(app_model, conversation_id, user)

        conversation.is_deleted = True
        conversation.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.session.commit()
