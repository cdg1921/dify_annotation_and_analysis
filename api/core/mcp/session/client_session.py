from datetime import timedelta
from typing import Any, Protocol

from pydantic import AnyUrl, TypeAdapter

from configs import dify_config
from core.mcp import types
from core.mcp.entities import SUPPORTED_PROTOCOL_VERSIONS, RequestContext
from core.mcp.session.base_session import BaseSession, RequestResponder

DEFAULT_CLIENT_INFO = types.Implementation(name="Dify", version=dify_config.project.version)

# cdg:SamplingFnT是采样函数的类型，用于处理采样请求。它是一个协议（Protocol），定义了__call__方法，该方法接受一个RequestContext和一个CreateMessageRequestParams，并返回一个CreateMessageResult或ErrorData。
class SamplingFnT(Protocol):
    def __call__(
        self,
        context: RequestContext["ClientSession", Any],
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData: ...
# cdg:最后的...表示该方法没有实现，类似于pass，只是为了满足协议的定义。在Python里，...是Ellipsis对象，常用于占位，表示“这里暂时省略实现”。
# cdg:这里的 ... 表示这个Protocol只是声明了call方法的签名，没有实现内容。这是一种类型提示写法，告诉类型检查器“实现这个 Protocol 的类/函数，必须有这个方法签名”。

class ListRootsFnT(Protocol):
    def __call__(self, context: RequestContext["ClientSession", Any]) -> types.ListRootsResult | types.ErrorData: ...


class LoggingFnT(Protocol):
    def __call__(
        self,
        params: types.LoggingMessageNotificationParams,
    ) -> None: ...


class MessageHandlerFnT(Protocol):
    def __call__(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None: ...


def _default_message_handler(
    message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
) -> None:
    if isinstance(message, Exception):
        raise ValueError(str(message))
    elif isinstance(message, (types.ServerNotification | RequestResponder)):
        pass


def _default_sampling_callback(
    context: RequestContext["ClientSession", Any],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="Sampling not supported",
    )


def _default_list_roots_callback(
    context: RequestContext["ClientSession", Any],
) -> types.ListRootsResult | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="List roots not supported",
    )


def _default_logging_callback(
    params: types.LoggingMessageNotificationParams,
) -> None:
    pass


ClientResponse: TypeAdapter[types.ClientResult | types.ErrorData] = TypeAdapter(types.ClientResult | types.ErrorData)

# cdg:客户端会话的类，继承自BaseSession，实现了客户端会话的功能
class ClientSession(
    BaseSession[
        types.ClientRequest,
        types.ClientNotification,
        types.ClientResult,
        types.ServerRequest,
        types.ServerNotification,
    ]
):
    def __init__(
        self,
        read_stream,
        write_stream,
        read_timeout_seconds: timedelta | None = None,
        sampling_callback: SamplingFnT | None = None,
        list_roots_callback: ListRootsFnT | None = None,
        logging_callback: LoggingFnT | None = None,
        message_handler: MessageHandlerFnT | None = None,
        client_info: types.Implementation | None = None,
    ) -> None:
        super().__init__(
            read_stream,
            write_stream,
            types.ServerRequest,
            types.ServerNotification,
            read_timeout_seconds=read_timeout_seconds,
        )
        self._client_info = client_info or DEFAULT_CLIENT_INFO
        self._sampling_callback = sampling_callback or _default_sampling_callback
        self._list_roots_callback = list_roots_callback or _default_list_roots_callback
        self._logging_callback = logging_callback or _default_logging_callback
        self._message_handler = message_handler or _default_message_handler

    # cdg:初始化方法，发送初始化请求，并返回初始化结果
    def initialize(self) -> types.InitializeResult:
        sampling = types.SamplingCapability()
        roots = types.RootsCapability(
            # TODO: Should this be based on whether we
            # _will_ send notifications, or only whether
            # they're supported?
            listChanged=True,
        )

        result = self.send_request(
            types.ClientRequest(
                types.InitializeRequest(
                    method="initialize",
                    params=types.InitializeRequestParams(
                        protocolVersion=types.LATEST_PROTOCOL_VERSION,
                        capabilities=types.ClientCapabilities(
                            sampling=sampling,
                            experimental=None,
                            roots=roots,
                        ),
                        clientInfo=self._client_info,
                    ),
                )
            ),
            types.InitializeResult,
        )

        if result.protocolVersion not in SUPPORTED_PROTOCOL_VERSIONS:
            raise RuntimeError(f"Unsupported protocol version from the server: {result.protocolVersion}")

        self.send_notification(
            types.ClientNotification(types.InitializedNotification(method="notifications/initialized"))
        )

        return result

    # cdg:发送ping请求，并返回空结果，用于检查服务器是否在线
    def send_ping(self) -> types.EmptyResult:
        """Send a ping request."""
        return self.send_request(
            types.ClientRequest(
                types.PingRequest(
                    method="ping",
                )
            ),
            types.EmptyResult,
        )

    # cdg:发送进度通知，用于通知客户端进度
    def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """Send a progress notification."""
        self.send_notification(
            types.ClientNotification(
                types.ProgressNotification(
                    method="notifications/progress",
                    params=types.ProgressNotificationParams(
                        progressToken=progress_token,
                        progress=progress,
                        total=total,
                    ),
                ),
            )
        )

    # cdg:发送设置日志级别请求，并返回空结果，用于设置日志级别
    def set_logging_level(self, level: types.LoggingLevel) -> types.EmptyResult:
        """Send a logging/setLevel request."""
        return self.send_request(
            types.ClientRequest(
                types.SetLevelRequest(
                    method="logging/setLevel",
                    params=types.SetLevelRequestParams(level=level),
                )
            ),
            types.EmptyResult,
        )

    # cdg:发送列表资源请求，并返回资源列表结果，用于获取资源列表
    def list_resources(self) -> types.ListResourcesResult:
        """Send a resources/list request."""
        return self.send_request(
            types.ClientRequest(
                types.ListResourcesRequest(
                    method="resources/list",
                )
            ),
            types.ListResourcesResult,
        )

    def list_resource_templates(self) -> types.ListResourceTemplatesResult:
        """Send a resources/templates/list request."""
        return self.send_request(
            types.ClientRequest(
                types.ListResourceTemplatesRequest(
                    method="resources/templates/list",
                )
            ),
            types.ListResourceTemplatesResult,
        )

    # cdg:发送读取资源请求，并返回资源内容结果，用于获取资源内容
    def read_resource(self, uri: AnyUrl) -> types.ReadResourceResult:
        """Send a resources/read request."""
        return self.send_request(
            types.ClientRequest(
                types.ReadResourceRequest(
                    method="resources/read",
                    params=types.ReadResourceRequestParams(uri=uri),
                )
            ),
            types.ReadResourceResult,
        )

    # cdg:发送订阅资源请求，并返回空结果
    def subscribe_resource(self, uri: AnyUrl) -> types.EmptyResult:
        """Send a resources/subscribe request."""
        return self.send_request(
            types.ClientRequest(
                types.SubscribeRequest(
                    method="resources/subscribe",
                    params=types.SubscribeRequestParams(uri=uri),
                )
            ),
            types.EmptyResult,
        )

    # cdg:发送取消订阅资源请求，并返回空结果
    def unsubscribe_resource(self, uri: AnyUrl) -> types.EmptyResult:
        """Send a resources/unsubscribe request."""
        return self.send_request(
            types.ClientRequest(
                types.UnsubscribeRequest(
                    method="resources/unsubscribe",
                    params=types.UnsubscribeRequestParams(uri=uri),
                )
            ),
            types.EmptyResult,
        )

    # cdg:发送调用工具请求，并返回工具调用结果，用于调用工具
    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> types.CallToolResult:
        """Send a tools/call request."""

        return self.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(name=name, arguments=arguments),
                )
            ),
            types.CallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
        )

    # cdg:发送列表提示请求，并返回提示列表结果，用于获取提示列表
    def list_prompts(self) -> types.ListPromptsResult:
        """Send a prompts/list request."""
        return self.send_request(
            types.ClientRequest(
                types.ListPromptsRequest(
                    method="prompts/list",
                )
            ),
            types.ListPromptsResult,
        )

    # cdg:发送获取提示请求，并返回提示结果，用于获取提示
    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
        """Send a prompts/get request."""
        return self.send_request(
            types.ClientRequest(
                types.GetPromptRequest(
                    method="prompts/get",
                    params=types.GetPromptRequestParams(name=name, arguments=arguments),
                )
            ),
            types.GetPromptResult,
        )

    # cdg:发送补全类请求，并返回补全结果
    def complete(
        self,
        ref: types.ResourceReference | types.PromptReference,
        argument: dict[str, str],
    ) -> types.CompleteResult:
        """Send a completion/complete request."""
        return self.send_request(
            types.ClientRequest(
                types.CompleteRequest(
                    method="completion/complete",
                    params=types.CompleteRequestParams(
                        ref=ref,
                        argument=types.CompletionArgument(**argument),
                    ),
                )
            ),
            types.CompleteResult,
        )

    # cdg:发送列表工具请求，并返回工具列表结果，用于获取工具列表
    def list_tools(self) -> types.ListToolsResult:
        """Send a tools/list request."""
        return self.send_request(
            types.ClientRequest(
                types.ListToolsRequest(
                    method="tools/list",
                )
            ),
            types.ListToolsResult,
        )

    # cdg:发送根列表改变通知，用于通知客户端根列表改变
    def send_roots_list_changed(self) -> None:
        """Send a roots/list_changed notification."""
        self.send_notification(
            types.ClientNotification(
                types.RootsListChangedNotification(
                    method="notifications/roots/list_changed",
                )
            )
        )

    # cdg:处理接收到的请求，根据请求类型调用相应的回调函数，并返回响应结果
    def _received_request(self, responder: RequestResponder[types.ServerRequest, types.ClientResult]) -> None:
        ctx = RequestContext[ClientSession, Any](
            request_id=responder.request_id,
            meta=responder.request_meta,
            session=self,
            lifespan_context=None,
        )

        match responder.request.root:
            case types.CreateMessageRequest(params=params):
                with responder:
                    response = self._sampling_callback(ctx, params)
                    client_response = ClientResponse.validate_python(response)
                    responder.respond(client_response)

            case types.ListRootsRequest():
                with responder:
                    list_roots_response = self._list_roots_callback(ctx)
                    client_response = ClientResponse.validate_python(list_roots_response)
                    responder.respond(client_response)

            case types.PingRequest():
                with responder:
                    return responder.respond(types.ClientResult(root=types.EmptyResult()))

    # cdg:处理接收到的通知，根据通知类型调用相应的回调函数，并返回响应结果
    def _handle_incoming(
        self,
        req: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None:
        """Handle incoming messages by forwarding to the message handler."""
        self._message_handler(req)

    def _received_notification(self, notification: types.ServerNotification) -> None:
        """Handle notifications from the server."""
        # Process specific notification types
        match notification.root:
            case types.LoggingMessageNotification(params=params):
                self._logging_callback(params)
            case _:
                pass
