from flask import Blueprint

from libs.external_api import ExternalApi

from .app.app_import import AppImportApi, AppImportConfirmApi
from .explore.audio import ChatAudioApi, ChatTextApi
from .explore.completion import ChatApi, ChatStopApi, CompletionApi, CompletionStopApi
from .explore.conversation import (
    ConversationApi,
    ConversationListApi,
    ConversationPinApi,
    ConversationRenameApi,
    ConversationUnPinApi,
)
from .explore.message import (
    MessageFeedbackApi,
    MessageListApi,
    MessageMoreLikeThisApi,
    MessageSuggestedQuestionApi,
)
from .explore.workflow import (
    InstalledAppWorkflowRunApi,
    InstalledAppWorkflowTaskStopApi,
)
from .files import FileApi, FilePreviewApi, FileSupportTypeApi
from .remote_files import RemoteFileInfoApi, RemoteFileUploadApi

# cdg:创建蓝图，用于处理console的请求。蓝图是Flask中的一个概念，其作用是将一个应用程序划分为多个模块，每个模块都有自己的路由、视图函数、模板等。
bp = Blueprint("console", __name__, url_prefix="/console/api")
api = ExternalApi(bp)

# cdg:文件相关接口路由，包括上传文件、预览文件、支持的文件类型查询等。
# File
api.add_resource(FileApi, "/files/upload")
api.add_resource(FilePreviewApi, "/files/<uuid:file_id>/preview")
api.add_resource(FileSupportTypeApi, "/files/support-type")

# cdg:远程文件相关接口路由，包括获取远程文件信息、上传远程文件等。
# Remote files
api.add_resource(RemoteFileInfoApi, "/remote-files/<path:url>")
api.add_resource(RemoteFileUploadApi, "/remote-files/upload")

# cdg:应用相关接口路由，包括导入应用、确认导入应用等。
# Import App
api.add_resource(AppImportApi, "/apps/imports")
api.add_resource(AppImportConfirmApi, "/apps/imports/<string:import_id>/confirm")

# Import other controllers
from . import admin, apikey, extension, feature, ping, setup, version

# Import app controllers
from .app import (
    advanced_prompt_template,
    agent,
    annotation,
    app,
    audio,
    completion,
    conversation,
    conversation_variables,
    generator,
    message,
    model_config,
    ops_trace,
    site,
    statistic,
    workflow,
    workflow_app_log,
    workflow_run,
    workflow_statistic,
)

# Import auth controllers
from .auth import activate, data_source_bearer_auth, data_source_oauth, forgot_password, login, oauth

# Import billing controllers
from .billing import billing

# Import datasets controllers
from .datasets import (
    data_source,
    datasets,
    datasets_document,
    datasets_segments,
    external,
    hit_testing,
    website,
)

# Import explore controllers
from .explore import (
    installed_app,
    parameter,
    recommended_app,
    saved_message,
)

# Explore Audio
api.add_resource(ChatAudioApi, "/installed-apps/<uuid:installed_app_id>/audio-to-text", endpoint="installed_app_audio")
api.add_resource(ChatTextApi, "/installed-apps/<uuid:installed_app_id>/text-to-audio", endpoint="installed_app_text")

# Explore Completion
api.add_resource(
    CompletionApi, "/installed-apps/<uuid:installed_app_id>/completion-messages", endpoint="installed_app_completion"
)
api.add_resource(
    CompletionStopApi,
    "/installed-apps/<uuid:installed_app_id>/completion-messages/<string:task_id>/stop",
    endpoint="installed_app_stop_completion",
)
api.add_resource(
    ChatApi, "/installed-apps/<uuid:installed_app_id>/chat-messages", endpoint="installed_app_chat_completion"
)
api.add_resource(
    ChatStopApi,
    "/installed-apps/<uuid:installed_app_id>/chat-messages/<string:task_id>/stop",
    endpoint="installed_app_stop_chat_completion",
)

# Explore Conversation
api.add_resource(
    ConversationRenameApi,
    "/installed-apps/<uuid:installed_app_id>/conversations/<uuid:c_id>/name",
    endpoint="installed_app_conversation_rename",
)
api.add_resource(
    ConversationListApi, "/installed-apps/<uuid:installed_app_id>/conversations", endpoint="installed_app_conversations"
)
api.add_resource(
    ConversationApi,
    "/installed-apps/<uuid:installed_app_id>/conversations/<uuid:c_id>",
    endpoint="installed_app_conversation",
)
api.add_resource(
    ConversationPinApi,
    "/installed-apps/<uuid:installed_app_id>/conversations/<uuid:c_id>/pin",
    endpoint="installed_app_conversation_pin",
)
api.add_resource(
    ConversationUnPinApi,
    "/installed-apps/<uuid:installed_app_id>/conversations/<uuid:c_id>/unpin",
    endpoint="installed_app_conversation_unpin",
)


# Explore Message
api.add_resource(MessageListApi, "/installed-apps/<uuid:installed_app_id>/messages", endpoint="installed_app_messages")
api.add_resource(
    MessageFeedbackApi,
    "/installed-apps/<uuid:installed_app_id>/messages/<uuid:message_id>/feedbacks",
    endpoint="installed_app_message_feedback",
)
api.add_resource(
    MessageMoreLikeThisApi,
    "/installed-apps/<uuid:installed_app_id>/messages/<uuid:message_id>/more-like-this",
    endpoint="installed_app_more_like_this",
)
api.add_resource(
    MessageSuggestedQuestionApi,
    "/installed-apps/<uuid:installed_app_id>/messages/<uuid:message_id>/suggested-questions",
    endpoint="installed_app_suggested_question",
)
# Explore Workflow
api.add_resource(InstalledAppWorkflowRunApi, "/installed-apps/<uuid:installed_app_id>/workflows/run")
api.add_resource(
    InstalledAppWorkflowTaskStopApi, "/installed-apps/<uuid:installed_app_id>/workflows/tasks/<string:task_id>/stop"
)

# Import tag controllers
from .tag import tags

# Import workspace controllers
from .workspace import account, load_balancing_config, members, model_providers, models, tool_providers, workspace
