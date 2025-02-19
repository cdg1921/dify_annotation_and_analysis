import json
from collections.abc import Generator
from typing import cast

from core.app.apps.base_app_generate_response_converter import AppGenerateResponseConverter
from core.app.entities.task_entities import (
    ErrorStreamResponse,
    NodeFinishStreamResponse,
    NodeStartStreamResponse,
    PingStreamResponse,
    WorkflowAppBlockingResponse,
    WorkflowAppStreamResponse,
)


# cdg:继承AppGenerateResponseConverter类，需要实现4种场景response输出方式：
# convert_blocking_full_response、convert_blocking_simple_response、
# convert_stream_full_response、convert_stream_simple_response
class WorkflowAppGenerateResponseConverter(AppGenerateResponseConverter):
    _blocking_response_type = WorkflowAppBlockingResponse

    @classmethod
    def convert_blocking_full_response(cls, blocking_response: WorkflowAppBlockingResponse) -> dict:  # type: ignore[override]
        """
        Convert blocking full response.
        :param blocking_response: blocking response
        :return:
        """
        # cdg:直接转成字典输出
        return dict(blocking_response.to_dict())

    @classmethod
    def convert_blocking_simple_response(cls, blocking_response: WorkflowAppBlockingResponse) -> dict:  # type: ignore[override]
        """
        Convert blocking simple response.
        :param blocking_response: blocking response
        :return:
        """
        # cdg:直接转成字典输出，full和simple的输出一样
        return cls.convert_blocking_full_response(blocking_response)

    @classmethod
    def convert_stream_full_response(
        cls,
        stream_response: Generator[WorkflowAppStreamResponse, None, None],  # type: ignore[override]
    ) -> Generator[str, None, None]:
        """
        Convert stream full response.
        :param stream_response: stream response
        :return:
        """
        for chunk in stream_response:
            chunk = cast(WorkflowAppStreamResponse, chunk)
            sub_stream_response = chunk.stream_response

            if isinstance(sub_stream_response, PingStreamResponse):
                yield "ping"
                continue

            # cdg:构建response_chunk的基础信息，包括事件类型、工作流运行ID，注意，此时还不包含生成的数据
            response_chunk = {
                "event": sub_stream_response.event.value,
                "workflow_run_id": chunk.workflow_run_id,
            }

            # cdg:将大模型生成的信息添加到response_chunk中，构成完整的输出
            if isinstance(sub_stream_response, ErrorStreamResponse):
                data = cls._error_to_stream_response(sub_stream_response.err)
                response_chunk.update(data)
            else:
                response_chunk.update(sub_stream_response.to_dict())
            yield json.dumps(response_chunk)

    @classmethod
    def convert_stream_simple_response(
        cls,
        stream_response: Generator[WorkflowAppStreamResponse, None, None],  # type: ignore[override]
    ) -> Generator[str, None, None]:
        """
        Convert stream simple response.
        :param stream_response: stream response
        :return:
        """
        for chunk in stream_response:
            chunk = cast(WorkflowAppStreamResponse, chunk)
            sub_stream_response = chunk.stream_response

            if isinstance(sub_stream_response, PingStreamResponse):
                yield "ping"
                continue

            # cdg:构建response_chunk的基础信息，包括事件类型、工作流运行ID，注意，此时还不包含生成的数据
            response_chunk = {
                "event": sub_stream_response.event.value,
                "workflow_run_id": chunk.workflow_run_id,
            }

            # cdg:将大模型生成的信息添加到response_chunk中，构成完整的输出
            if isinstance(sub_stream_response, ErrorStreamResponse):
                data = cls._error_to_stream_response(sub_stream_response.err)
                response_chunk.update(data)
            elif isinstance(sub_stream_response, NodeStartStreamResponse | NodeFinishStreamResponse):
                response_chunk.update(sub_stream_response.to_ignore_detail_dict())
            else:
                response_chunk.update(sub_stream_response.to_dict())

            # cdg:输出结果为字符串
            yield json.dumps(response_chunk)
