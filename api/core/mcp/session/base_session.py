import logging
import queue
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import timedelta
from types import TracebackType
from typing import Any, Generic, Self, TypeVar

from httpx import HTTPStatusError
from pydantic import BaseModel

from core.mcp.error import MCPAuthError, MCPConnectionError
from core.mcp.types import (
    CancelledNotification,
    ClientNotification,
    ClientRequest,
    ClientResult,
    ErrorData,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MessageMetadata,
    RequestId,
    RequestParams,
    ServerMessageMetadata,
    ServerNotification,
    ServerRequest,
    ServerResult,
    SessionMessage,
)

# cdg:TypeVar是类型变量，用于定义泛型类型。它允许你在定义类或函数时使用类型参数，这样可以让类或函数在处理不同类型的数据时更加灵活和类型安全。
# cdg:bound=BaseModel，说明ReceiveResultT必须继承自BaseModel。
# cdg:SendResultT可以是ClientResult或ServerResult类型
# cdg:ReceiveRequestT可以是ClientRequest或ServerRequest类型
# cdg:泛型类型的名字最后加T，T是TypeVar的缩写，这是DIFY的命名规范。
SendRequestT = TypeVar("SendRequestT", ClientRequest, ServerRequest)
SendResultT = TypeVar("SendResultT", ClientResult, ServerResult)
SendNotificationT = TypeVar("SendNotificationT", ClientNotification, ServerNotification)
ReceiveRequestT = TypeVar("ReceiveRequestT", ClientRequest, ServerRequest)
ReceiveResultT = TypeVar("ReceiveResultT", bound=BaseModel)
ReceiveNotificationT = TypeVar("ReceiveNotificationT", ClientNotification, ServerNotification)
DEFAULT_RESPONSE_READ_TIMEOUT = 1.0

# cdg:RequestResponder是请求响应器，用于处理请求和响应。它是一个上下文管理器，确保请求的正确清理和取消处理。
# cdg:这个类继承自Generic[ReceiveRequestT, SendResultT]，说明它是一个泛型类（generic class）。泛型类允许你在使用这个类时指定类型参数，这样可以让类在处理不同类型的数据时更加灵活和类型安全。
class RequestResponder(Generic[ReceiveRequestT, SendResultT]):
    """Handles responding to MCP requests and manages request lifecycle.

    This class MUST be used as a context manager to ensure proper cleanup and
    cancellation handling:

    Example:
        with request_responder as resp:
            resp.respond(result)

    The context manager ensures:
    1. Proper cancellation scope setup and cleanup
    2. Request completion tracking
    3. Cleanup of in-flight requests
    """

    request: ReceiveRequestT
    _session: Any
    # cdg:on_complete是回调函数，用于处理请求完成后的操作。它是一个回调函数，用于处理请求完成后的操作。
    # cdg:这里的RequestResponder用双引号是为了前向引用（forward reference）。
    # cdg:在Python的类型注解中，如果你要在类的属性或方法的类型注解里引用当前类本身（即类还没完全定义好），直接写类名会报错，因为此时类还没被解释器“看到”。
    # cdg:用字符串包裹类名（如 "RequestResponder[ReceiveRequestT, SendResultT]"）可以让解释器先跳过类型检查，等到类型检查器（如 mypy）或IDE静态分析时再解析这个类型。
    # cdg:Python 3.7+ 支持from __future__ import annotations后可以不用引号，但加引号兼容性更好。
    _on_complete: Callable[["RequestResponder[ReceiveRequestT, SendResultT]"], Any]

    def __init__(
        self,
        request_id: RequestId,
        request_meta: RequestParams.Meta | None,
        request: ReceiveRequestT,
        session: """BaseSession[
            SendRequestT,
            SendNotificationT,
            SendResultT,
            ReceiveRequestT,
            ReceiveNotificationT
        ]""",
        on_complete: Callable[["RequestResponder[ReceiveRequestT, SendResultT]"], Any],
    ) -> None:
        self.request_id = request_id
        self.request_meta = request_meta
        self.request = request
        self._session = session
        self._completed = False
        self._on_complete = on_complete
        self._entered = False  # Track if we're in a context manager

    def __enter__(self) -> "RequestResponder[ReceiveRequestT, SendResultT]":
        """Enter the context manager, enabling request cancellation tracking."""
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, performing cleanup and notifying completion."""
        try:
            if self._completed:
                self._on_complete(self)
        finally:
            self._entered = False

    def respond(self, response: SendResultT | ErrorData) -> None:
        """Send a response for this request.

        Must be called within a context manager block.
        Raises:
            RuntimeError: If not used within a context manager
            AssertionError: If request was already responded to
        """
        if not self._entered:
            raise RuntimeError("RequestResponder must be used as a context manager")
        assert not self._completed, "Request already responded to"

        self._completed = True

        self._session._send_response(request_id=self.request_id, response=response)

    def cancel(self) -> None:
        """Cancel this request and mark it as completed."""
        if not self._entered:
            raise RuntimeError("RequestResponder must be used as a context manager")

        self._completed = True  # Mark as completed so it's removed from in_flight
        # Send an error response to indicate cancellation
        self._session._send_response(
            request_id=self.request_id,
            response=ErrorData(code=0, message="Request cancelled", data=None),
        )


class BaseSession(
    Generic[
        SendRequestT,
        SendNotificationT,
        SendResultT,
        ReceiveRequestT,
        ReceiveNotificationT,
    ],
):
    """
    Implements an MCP "session" on top of read/write streams, including features
    like request/response linking, notifications, and progress.

    This class is a context manager that automatically starts processing
    messages when entered.
    """

    _response_streams: dict[RequestId, queue.Queue[JSONRPCResponse | JSONRPCError]]
    _request_id: int
    _in_flight: dict[RequestId, RequestResponder[ReceiveRequestT, SendResultT]]
    _receive_request_type: type[ReceiveRequestT]
    _receive_notification_type: type[ReceiveNotificationT]

    def __init__(
        self,
        read_stream: queue.Queue,
        write_stream: queue.Queue,
        receive_request_type: type[ReceiveRequestT],
        receive_notification_type: type[ReceiveNotificationT],
        # If none, reading will never time out
        read_timeout_seconds: timedelta | None = None,
    ) -> None:
        self._read_stream = read_stream
        self._write_stream = write_stream
        self._response_streams = {}
        self._request_id = 0
        self._receive_request_type = receive_request_type
        self._receive_notification_type = receive_notification_type
        self._session_read_timeout_seconds = read_timeout_seconds
        self._in_flight = {}
        self._exit_stack = ExitStack()

    # cdg: 在 Python类中，以“__”为名称首尾的函数被称为魔术方法（Magic Methods）或特殊方法（Special Methods），比如 __init__、__enter__、__exit__、__str__、__repr__ 等。这些方法和普通方法有以下主要区别：
    # cdg: 1、触发方式不同
    # cdg: 魔术方法：不是直接调用的，而是由Python解释器在特定场景下自动调用。例如：
    # cdg: __init__ 在对象创建时自动调用（如obj = MyClass()时）。
    # cdg: __enter__和 __exit__在使用with语句时自动调用（如with obj as x:）。
    # cdg: __str__ 在使用str(obj)或print(obj)时自动调用。
    # cdg: 2、命名方式不同
    # cdg: 魔术方法：通常以双下划线开头和结尾，如 __init__、__enter__、__exit__。
    # cdg: 普通方法：直接使用方法名，如 def my_method(self):。
    # cdg: 3、用途不同
    # cdg: 魔术方法：用于实现特定功能，如初始化、上下文管理、序列化等。
    # cdg: 普通方法：用于实现类的业务逻辑。
    # cdg: 4、参数不同
    # cdg: 魔术方法：通常没有参数，或者参数是固定的。
    # cdg: 普通方法：可以有参数，参数可以是任意的。

    # cdg:上下文管理器，用于初始化线程池和接收循环。
    # cdg:Self是类型提示，表示返回当前类的实例。
    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor()
        self._receiver_future = self._executor.submit(self._receive_loop)
        return self

    # cdg:检查接收器状态，如果接收器已完成，则获取结果。
    def check_receiver_status(self) -> None:
        if self._receiver_future.done():
            self._receiver_future.result()

    # cdg:退出上下文管理器，用于关闭线程池和接收循环。
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self._exit_stack.close()
        self._read_stream.put(None)
        self._write_stream.put(None)

    # cdg:发送请求，并等待响应。如果请求读取超时，则优先于会话读取超时时间。
    def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
        request_read_timeout_seconds: timedelta | None = None,
        metadata: MessageMetadata = None,
    ) -> ReceiveResultT:
        """
        Sends a request and wait for a response. Raises an McpError if the
        response contains an error. If a request read timeout is provided, it
        will take precedence over the session read timeout.

        Do not use this method to emit notifications! Use send_notification()
        instead.
        """
        self.check_receiver_status()

        request_id = self._request_id
        self._request_id = request_id + 1

        response_queue: queue.Queue[JSONRPCResponse | JSONRPCError] = queue.Queue()
        self._response_streams[request_id] = response_queue

        try:
            jsonrpc_request = JSONRPCRequest(
                jsonrpc="2.0",
                id=request_id,
                **request.model_dump(by_alias=True, mode="json", exclude_none=True),
            )

            self._write_stream.put(SessionMessage(message=JSONRPCMessage(jsonrpc_request), metadata=metadata))
            timeout = DEFAULT_RESPONSE_READ_TIMEOUT
            if request_read_timeout_seconds is not None:
                timeout = float(request_read_timeout_seconds.total_seconds())
            elif self._session_read_timeout_seconds is not None:
                timeout = float(self._session_read_timeout_seconds.total_seconds())
            while True:
                try:
                    # cdg:从响应队列中获取响应或错误，如果超时，则检查接收器状态，如果队列为空，则继续循环
                    response_or_error = response_queue.get(timeout=timeout)
                    break
                except queue.Empty:
                    self.check_receiver_status()
                    continue

            if response_or_error is None:
                raise MCPConnectionError(
                    ErrorData(
                        code=500,
                        message="No response received",
                    )
                )
            elif isinstance(response_or_error, JSONRPCError):
                if response_or_error.error.code == 401:
                    raise MCPAuthError(
                        ErrorData(code=response_or_error.error.code, message=response_or_error.error.message)
                    )
                else:
                    raise MCPConnectionError(
                        ErrorData(code=response_or_error.error.code, message=response_or_error.error.message)
                    )
            else:
                return result_type.model_validate(response_or_error.result)

        finally:
            self._response_streams.pop(request_id, None)

    # cdg:发送通知，不期望响应。
    def send_notification(
        self,
        notification: SendNotificationT,
        related_request_id: RequestId | None = None,
    ) -> None:
        """
        Emits a notification, which is a one-way message that does not expect
        a response.
        """
        self.check_receiver_status()

        # Some transport implementations may need to set the related_request_id
        # to attribute to the notifications to the request that triggered them.
        jsonrpc_notification = JSONRPCNotification(
            jsonrpc="2.0",
            **notification.model_dump(by_alias=True, mode="json", exclude_none=True),
        )
        session_message = SessionMessage(
            message=JSONRPCMessage(jsonrpc_notification),
            metadata=ServerMessageMetadata(related_request_id=related_request_id) if related_request_id else None,
        )
        self._write_stream.put(session_message)

    # cdg:发送响应，用于发送响应消息。
    def _send_response(self, request_id: RequestId, response: SendResultT | ErrorData) -> None:
        if isinstance(response, ErrorData):
            jsonrpc_error = JSONRPCError(jsonrpc="2.0", id=request_id, error=response)
            session_message = SessionMessage(message=JSONRPCMessage(jsonrpc_error))
            self._write_stream.put(session_message)
        else:
            jsonrpc_response = JSONRPCResponse(
                jsonrpc="2.0",
                id=request_id,
                result=response.model_dump(by_alias=True, mode="json", exclude_none=True),
            )
            session_message = SessionMessage(message=JSONRPCMessage(jsonrpc_response))
            self._write_stream.put(session_message)

    # cdg:接收循环，用于接收消息。
    def _receive_loop(self) -> None:
        """
        Main message processing loop.
        In a real synchronous implementation, this would likely run in a separate thread.
        """
        while True:
            try:
                # Attempt to receive a message (this would be blocking in a synchronous context)
                message = self._read_stream.get(timeout=DEFAULT_RESPONSE_READ_TIMEOUT)
                if message is None:
                    break
                if isinstance(message, HTTPStatusError):
                    response_queue = self._response_streams.get(self._request_id - 1)
                    if response_queue is not None:
                        response_queue.put(
                            JSONRPCError(
                                jsonrpc="2.0",
                                id=self._request_id - 1,
                                error=ErrorData(code=message.response.status_code, message=message.args[0]),
                            )
                        )
                    else:
                        self._handle_incoming(RuntimeError(f"Received response with an unknown request ID: {message}"))
                elif isinstance(message, Exception):
                    self._handle_incoming(message)
                elif isinstance(message.message.root, JSONRPCRequest):
                    validated_request = self._receive_request_type.model_validate(
                        message.message.root.model_dump(by_alias=True, mode="json", exclude_none=True)
                    )

                    responder = RequestResponder(
                        request_id=message.message.root.id,
                        request_meta=validated_request.root.params.meta if validated_request.root.params else None,
                        request=validated_request,
                        session=self,
                        on_complete=lambda r: self._in_flight.pop(r.request_id, None),
                    )

                    self._in_flight[responder.request_id] = responder
                    self._received_request(responder)

                    if not responder._completed:
                        self._handle_incoming(responder)

                elif isinstance(message.message.root, JSONRPCNotification):
                    try:
                        notification = self._receive_notification_type.model_validate(
                            message.message.root.model_dump(by_alias=True, mode="json", exclude_none=True)
                        )
                        # Handle cancellation notifications
                        if isinstance(notification.root, CancelledNotification):
                            cancelled_id = notification.root.params.requestId
                            if cancelled_id in self._in_flight:
                                self._in_flight[cancelled_id].cancel()
                        else:
                            self._received_notification(notification)
                            self._handle_incoming(notification)
                    except Exception as e:
                        # For other validation errors, log and continue
                        logging.warning(f"Failed to validate notification: {e}. Message was: {message.message.root}")
                else:  # Response or error
                    response_queue = self._response_streams.get(message.message.root.id)
                    if response_queue is not None:
                        response_queue.put(message.message.root)
                    else:
                        self._handle_incoming(RuntimeError(f"Server Error: {message}"))
            except queue.Empty:
                continue
            except Exception as e:
                logging.exception("Error in message processing loop")
                raise

    # cdg:处理接收到的请求。
    def _received_request(self, responder: RequestResponder[ReceiveRequestT, SendResultT]) -> None:
        """
        Can be overridden by subclasses to handle a request without needing to
        listen on the message stream.

        If the request is responded to within this method, it will not be
        forwarded on to the message stream.
        """
        pass

    # cdg:处理接收到的通知。
    def _received_notification(self, notification: ReceiveNotificationT) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        pass

    # cdg:发送进度通知，用于发送进度消息。
    def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Sends a progress notification for a request that is currently being
        processed.
        """
        pass

    # cdg:处理接收到的消息。
    def _handle_incoming(
        self,
        req: RequestResponder[ReceiveRequestT, SendResultT] | ReceiveNotificationT | Exception,
    ) -> None:
        """A generic handler for incoming messages. Overwritten by subclasses."""
        pass
