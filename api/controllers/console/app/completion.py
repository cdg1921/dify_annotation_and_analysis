import logging

import flask_login  # type: ignore
from flask_restful import Resource, reqparse  # type: ignore
from werkzeug.exceptions import InternalServerError, NotFound

import services
from controllers.console import api
from controllers.console.app.error import (
    AppUnavailableError,
    CompletionRequestError,
    ConversationCompletedError,
    ProviderModelCurrentlyNotSupportError,
    ProviderNotInitializeError,
    ProviderQuotaExceededError,
)
from controllers.console.app.wraps import get_app_model
from controllers.console.wraps import account_initialization_required, setup_required
from controllers.web.error import InvokeRateLimitError as InvokeRateLimitHttpError
from core.app.apps.base_app_queue_manager import AppQueueManager
from core.app.entities.app_invoke_entities import InvokeFrom
from core.errors.error import (
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from core.model_runtime.errors.invoke import InvokeError
from libs import helper
from libs.helper import uuid_value
from libs.login import login_required
from models.model import AppMode
from services.app_generate_service import AppGenerateService
from services.errors.llm import InvokeRateLimitError

# cdg:定义CompletionMessageApi，用于定义CompletionMessageApi
# define completion message api for user
class CompletionMessageApi(Resource):
    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=AppMode.COMPLETION)
    def post(self, app_model):
        # cdg:获取请求参数，包括inputs、query、files、model_config、response_mode、retriever_from
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", type=dict, required=True, location="json")
        parser.add_argument("query", type=str, location="json", default="")
        parser.add_argument("files", type=list, required=False, location="json")
        parser.add_argument("model_config", type=dict, required=True, location="json")
        parser.add_argument("response_mode", type=str, choices=["blocking", "streaming"], location="json")
        parser.add_argument("retriever_from", type=str, required=False, default="dev", location="json")
        args = parser.parse_args()

        # cdg:获取响应模式，如果响应模式为blocking，则设置为False，否则设置为True
        streaming = args["response_mode"] != "blocking"
        args["auto_generate_name"] = False

        # cdg:获取当前用户
        account = flask_login.current_user

        # cdg:生成响应
        try:
            response = AppGenerateService.generate(
                app_model=app_model, user=account, args=args, invoke_from=InvokeFrom.DEBUGGER, streaming=streaming
            )
            # cdg:响应消息后处理，返回简洁的响应
            return helper.compact_generate_response(response)
        # cdg:如果会话不存在，则抛出异常
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logging.exception("App model config broken.")
            raise AppUnavailableError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise e
        except Exception as e:
            logging.exception("internal server error.")
            raise InternalServerError()

# cdg:定义CompletionMessageStopApi，实现CompletionMessage生成过程的停止功能
class CompletionMessageStopApi(Resource):
    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=AppMode.COMPLETION)
    def post(self, app_model, task_id):
        account = flask_login.current_user
        # cdg:设置停止标志，模型生成过程中检测到停止标志后，会停止生成
        AppQueueManager.set_stop_flag(task_id, InvokeFrom.DEBUGGER, account.id)

        return {"result": "success"}, 200

# cdg:定义ChatMessageApi，实现对话的生成过程
class ChatMessageApi(Resource):
    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=[AppMode.CHAT, AppMode.AGENT_CHAT])
    def post(self, app_model):
        # cdg:获取请求参数，与上面的CompletionMessageApi相比，增加了conversation_id、parent_message_id，实现多轮对话
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", type=dict, required=True, location="json")
        parser.add_argument("query", type=str, required=True, location="json")
        parser.add_argument("files", type=list, required=False, location="json")
        parser.add_argument("model_config", type=dict, required=True, location="json")
        parser.add_argument("conversation_id", type=uuid_value, location="json")
        parser.add_argument("parent_message_id", type=uuid_value, required=False, location="json")
        parser.add_argument("response_mode", type=str, choices=["blocking", "streaming"], location="json")
        parser.add_argument("retriever_from", type=str, required=False, default="dev", location="json")
        args = parser.parse_args()

        streaming = args["response_mode"] != "blocking"
        args["auto_generate_name"] = False

        account = flask_login.current_user

        try:
            # cdg:生成响应，与上面的CompletionMessageApi相比，调用了相同的生成方法，只是传入的参数不同
            response = AppGenerateService.generate(
                app_model=app_model, user=account, args=args, invoke_from=InvokeFrom.DEBUGGER, streaming=streaming
            )
            # cdg:响应消息后处理，返回简洁的响应
            return helper.compact_generate_response(response)
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logging.exception("App model config broken.")
            raise AppUnavailableError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except InvokeRateLimitError as ex:
            raise InvokeRateLimitHttpError(ex.description)
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise e
        except Exception as e:
            logging.exception("internal server error.")
            raise InternalServerError()

# cdg:定义ChatMessageStopApi，实现对话的生成过程的停止功能
class ChatMessageStopApi(Resource):
    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=[AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.ADVANCED_CHAT])
    def post(self, app_model, task_id):
        account = flask_login.current_user

        AppQueueManager.set_stop_flag(task_id, InvokeFrom.DEBUGGER, account.id)

        return {"result": "success"}, 200

# cdg:定义CompletionMessageApi、CompletionMessageStopApi、ChatMessageApi、ChatMessageStopApi的路由
api.add_resource(CompletionMessageApi, "/apps/<uuid:app_id>/completion-messages")
api.add_resource(CompletionMessageStopApi, "/apps/<uuid:app_id>/completion-messages/<string:task_id>/stop")
api.add_resource(ChatMessageApi, "/apps/<uuid:app_id>/chat-messages")
api.add_resource(ChatMessageStopApi, "/apps/<uuid:app_id>/chat-messages/<string:task_id>/stop")
