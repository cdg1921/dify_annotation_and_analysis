import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from typing import Any, Union

from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.entities.task_entities import AppBlockingResponse, AppStreamResponse
from core.errors.error import ModelCurrentlyNotSupportError, ProviderTokenNotInitError, QuotaExceededError
from core.model_runtime.errors.invoke import InvokeError

# cdg:[
#     AdvancedChatAppGenerateResponseConverter、
#     AgentChatAppGenerateResponseConverter、
#     ChatAppGenerateResponseConverter、
#     CompletionAppGenerateResponseConverter、
#     WorkflowAppGenerateResponseConverter
# ]
# -> AppGenerateResponseConverter
class AppGenerateResponseConverter(ABC):
    _blocking_response_type: type[AppBlockingResponse]

    @classmethod
    def convert(
        cls,
        response: Union[AppBlockingResponse, Generator[AppStreamResponse, Any, None]],
        invoke_from: InvokeFrom,
    ) -> Mapping[str, Any] | Generator[str, None, None]:
        # cdg:不同调用来源，输出方式不一样，debug和service_api方式，输出全部信息；其他方式，输出简化格式信息
        # cdg:4中场景，blocking或stream、full或simple
        if invoke_from in {InvokeFrom.DEBUGGER, InvokeFrom.SERVICE_API}:
            if isinstance(response, AppBlockingResponse):
                # cdg:转为字典格式输出
                return cls.convert_blocking_full_response(response)
            else:
                # cdg:流式输出
                def _generate_full_response() -> Generator[str, Any, None]:
                    for chunk in cls.convert_stream_full_response(response):
                        if chunk == "ping":
                            yield f"event: {chunk}\n\n"
                        else:
                            yield f"data: {chunk}\n\n"

                return _generate_full_response()
        else:
            if isinstance(response, AppBlockingResponse):
                # cdg:转为字典格式输出，添加metadata信息
                return cls.convert_blocking_simple_response(response)
            else:

                def _generate_simple_response() -> Generator[str, Any, None]:
                    for chunk in cls.convert_stream_simple_response(response):
                        if chunk == "ping":
                            yield f"event: {chunk}\n\n"
                        else:
                            yield f"data: {chunk}\n\n"

                return _generate_simple_response()

    @classmethod
    @abstractmethod
    def convert_blocking_full_response(cls, blocking_response: AppBlockingResponse) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_blocking_simple_response(cls, blocking_response: AppBlockingResponse) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_stream_full_response(
        cls, stream_response: Generator[AppStreamResponse, None, None]
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_stream_simple_response(
        cls, stream_response: Generator[AppStreamResponse, None, None]
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    @classmethod
    def _get_simple_metadata(cls, metadata: dict[str, Any]):
        """
        Get simple metadata.
        :param metadata: metadata
        :return:
        """
        # show_retrieve_source
        updated_resources = []
        if "retriever_resources" in metadata:
            for resource in metadata["retriever_resources"]:
                updated_resources.append(
                    {
                        "segment_id": resource.get("segment_id", ""),
                        "position": resource["position"],
                        "document_name": resource["document_name"],
                        "score": resource["score"],
                        "content": resource["content"],
                    }
                )
            metadata["retriever_resources"] = updated_resources

        # cdg:删除注释信息
        # show annotation reply
        if "annotation_reply" in metadata:
            del metadata["annotation_reply"]

        # cdg:删除usage计费信息
        # show usage
        if "usage" in metadata:
            del metadata["usage"]

        return metadata

    @classmethod
    def _error_to_stream_response(cls, e: Exception) -> dict:
        """
        Error to stream response.
        :param e: exception
        :return:
        """
        error_responses = {
            ValueError: {"code": "invalid_param", "status": 400},
            ProviderTokenNotInitError: {"code": "provider_not_initialize", "status": 400},
            QuotaExceededError: {
                "code": "provider_quota_exceeded",
                "message": "Your quota for Dify Hosted Model Provider has been exhausted. "
                "Please go to Settings -> Model Provider to complete your own provider credentials.",
                "status": 400,
            },
            ModelCurrentlyNotSupportError: {"code": "model_currently_not_support", "status": 400},
            InvokeError: {"code": "completion_request_error", "status": 400},
        }

        # Determine the response based on the type of exception
        data = None
        for k, v in error_responses.items():
            if isinstance(e, k):
                data = v

        if data:
            data.setdefault("message", getattr(e, "description", str(e)))
        else:
            logging.error(e)
            data = {
                "code": "internal_server_error",
                "message": "Internal Server Error, please contact support.",
                "status": 500,
            }

        return data
