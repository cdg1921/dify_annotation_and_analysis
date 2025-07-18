from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, FileUrl, RootModel
from pydantic.networks import AnyUrl, UrlConstraints

"""
Model Context Protocol bindings for Python

These bindings were generated from https://github.com/modelcontextprotocol/specification,
using Claude, with a prompt something like the following:

Generate idiomatic Python bindings for this schema for MCP, or the "Model Context
Protocol." The schema is defined in TypeScript, but there's also a JSON Schema version
for reference.

* For the bindings, let's use Pydantic V2 models.
* Each model should allow extra fields everywhere, by specifying `model_config =
  ConfigDict(extra='allow')`. Do this in every case, instead of a custom base class.
* Union types should be represented with a Pydantic `RootModel`.
* Define additional model classes instead of using dictionaries. Do this even if they're
  not separate types in the schema.

# cdg:
Model Context Protocol 的Python绑定
这些绑定是从 https://github.com/modelcontextprotocol/specification生成的，
使用Claude，并且提示语大致如下：
为MCP（即“模型上下文协议”）的这个schema生成地道的Python绑定。schema是用TypeScript定义的，但也有JSON Schema版本可供参考。
绑定中，使用Pydantic V2的模型。
每个模型都应允许额外字段，通过指定model_config = ConfigDict(extra='allow')实现。每个模型都要这样做，而不是用自定义基类。
联合类型应使用Pydantic的RootModel表示。
应该定义额外的模型类，而不是直接用字典。即使schema里不是单独的类型，也要这样做。
"""
# Client support both version, not support 2025-06-18 yet.
LATEST_PROTOCOL_VERSION = "2025-03-26"
# Server support 2024-11-05 to allow claude to use.
SERVER_LATEST_PROTOCOL_VERSION = "2024-11-05"
# cdg:ProgressToken是一个类型别名（type alias），它可以是 str（字符串）类型，也可以是 int（整数）类型。
# cdg:在Python 3.10及以上版本中，|被用作类型联合（Union）的简写。例如，str | int 等价于 Union[str, int]，表示变量可以是字符串或整数。
ProgressToken = str | int  
Cursor = str
# cdg:Role是一个类型别名（type alias），它可以是 "user"（用户）类型，也可以是 "assistant"（助手）类型。
Role = Literal["user", "assistant"] 
# cdg:Literal和Optional的区别在于：Literal 用于限制参数或变量的值只能是指定的几个常量之一，常用于枚举、状态等场景。而Optional 用于表示参数或变量可以是None或指定的类型。
RequestId = Annotated[int | str, Field(union_mode="left_to_right")]
AnyFunction: TypeAlias = Callable[..., Any]

# cdg:定义请求参数的模型。
class RequestParams(BaseModel):
    class Meta(BaseModel):
        progressToken: ProgressToken | None = None
        """
        If specified, the caller requests out-of-band progress notifications for
        this request (as represented by notifications/progress). The value of this
        parameter is an opaque token that will be attached to any subsequent
        notifications. The receiver is not obligated to provide these notifications.
        """

        model_config = ConfigDict(extra="allow")

    meta: Meta | None = Field(alias="_meta", default=None)

# cdg:定义通知参数的模型。
class NotificationParams(BaseModel):
    class Meta(BaseModel):
        model_config = ConfigDict(extra="allow")

    meta: Meta | None = Field(alias="_meta", default=None)
    """
    This parameter name is reserved by MCP to allow clients and servers to attach
    additional metadata to their notifications.
    """

# cdg:定义请求参数、通知参数、方法的类型。TypeVar是类型变量，用于定义泛型类型。
RequestParamsT = TypeVar("RequestParamsT", bound=RequestParams | dict[str, Any] | None)
NotificationParamsT = TypeVar("NotificationParamsT", bound=NotificationParams | dict[str, Any] | None)
MethodT = TypeVar("MethodT", bound=str)

# cdg:定义请求的模型。
class Request(BaseModel, Generic[RequestParamsT, MethodT]):
    """Base class for JSON-RPC requests."""

    method: MethodT
    params: RequestParamsT
    model_config = ConfigDict(extra="allow")

# cdg:定义分页请求的模型。
class PaginatedRequest(Request[RequestParamsT, MethodT]):
    cursor: Cursor | None = None
    """
    An opaque token representing the current pagination position.
    If provided, the server should return results starting after this cursor.
    """

# cdg:定义通知的模型。
class Notification(BaseModel, Generic[NotificationParamsT, MethodT]):
    """Base class for JSON-RPC notifications."""

    method: MethodT
    params: NotificationParamsT
    model_config = ConfigDict(extra="allow")

# cdg:定义结果的模型。
class Result(BaseModel):
    """Base class for JSON-RPC results."""

    model_config = ConfigDict(extra="allow")

    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    This result property is reserved by the protocol to allow clients and servers to
    attach additional metadata to their responses.
    """

# cdg:定义分页结果的模型。
class PaginatedResult(Result):
    nextCursor: Cursor | None = None
    """
    An opaque token representing the pagination position after the last returned result.
    If present, there may be more results available.
    """

# cdg:定义JSON-RPC请求的模型。
class JSONRPCRequest(Request[dict[str, Any] | None, str]):
    """A request that expects a response."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    method: str
    params: dict[str, Any] | None = None

# cdg:定义JSON-RPC通知的模型。
class JSONRPCNotification(Notification[dict[str, Any] | None, str]):
    """A notification which does not expect a response."""

    jsonrpc: Literal["2.0"]
    params: dict[str, Any] | None = None

# cdg:定义JSON-RPC响应的模型。  
class JSONRPCResponse(BaseModel):
    """A successful (non-error) response to a request."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    result: dict[str, Any]
    model_config = ConfigDict(extra="allow")

# cdg:JSON-RPC错误码
# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# cdg:定义JSON-RPC错误信息的模型。
class ErrorData(BaseModel):
    """Error information for JSON-RPC error responses."""

    code: int
    """The error type that occurred."""

    message: str
    """
    A short description of the error. The message SHOULD be limited to a concise single
    sentence.
    """

    data: Any | None = None
    """
    Additional information about the error. The value of this member is defined by the
    sender (e.g. detailed error information, nested errors etc.).
    """

    model_config = ConfigDict(extra="allow")

# cdg:定义JSON-RPC错误的模型。
class JSONRPCError(BaseModel):
    """A response to a request that indicates an error occurred."""

    jsonrpc: Literal["2.0"]
    id: str | int
    error: ErrorData
    model_config = ConfigDict(extra="allow")

# cdg:定义JSON-RPC消息的模型。RootModel是Pydantic的模型，用于表示联合类型，意思是JSONRPCMessage可以是JSONRPCRequest、JSONRPCNotification、JSONRPCResponse、JSONRPCError中的一个。
class JSONRPCMessage(RootModel[JSONRPCRequest | JSONRPCNotification | JSONRPCResponse | JSONRPCError]):
    pass

# cdg:定义空结果的模型。
class EmptyResult(Result):
    """A response that indicates success but carries no data."""

# cdg:定义实现模型的模型。
class Implementation(BaseModel):
    """Describes the name and version of an MCP implementation."""

    name: str
    version: str
    model_config = ConfigDict(extra="allow")

# cdg:定义根操作的模型。
class RootsCapability(BaseModel):
    """Capability for root operations."""

    listChanged: bool | None = None
    """Whether the client supports notifications for changes to the roots list."""
    model_config = ConfigDict(extra="allow")

# cdg:定义采样操作的模型。
class SamplingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")

# cdg:定义客户端能力的模型。
class ClientCapabilities(BaseModel):
    """Capabilities a client may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the client supports."""
    sampling: SamplingCapability | None = None
    """Present if the client supports sampling from an LLM."""
    roots: RootsCapability | None = None
    """Present if the client supports listing roots."""
    model_config = ConfigDict(extra="allow")

# cdg:定义提示操作的模型。
class PromptsCapability(BaseModel):
    """Capability for prompts operations."""

    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the prompt list."""
    model_config = ConfigDict(extra="allow")

# cdg:定义资源操作的模型。
class ResourcesCapability(BaseModel):
    """Capability for resources operations."""

    subscribe: bool | None = None
    """Whether this server supports subscribing to resource updates."""
    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the resource list."""
    model_config = ConfigDict(extra="allow")

# cdg:定义工具操作的模型。
class ToolsCapability(BaseModel):
    """Capability for tools operations."""

    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the tool list."""
    model_config = ConfigDict(extra="allow")

# cdg:定义日志操作的模型。
class LoggingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")

# cdg:定义服务器能力的模型。
class ServerCapabilities(BaseModel):
    """Capabilities that a server may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the server supports."""
    logging: LoggingCapability | None = None
    """Present if the server supports sending log messages to the client."""
    prompts: PromptsCapability | None = None
    """Present if the server offers any prompt templates."""
    resources: ResourcesCapability | None = None
    """Present if the server offers any resources to read."""
    tools: ToolsCapability | None = None
    """Present if the server offers any tools to call."""
    model_config = ConfigDict(extra="allow")

# cdg:定义初始化请求参数的模型。
class InitializeRequestParams(RequestParams):
    """Parameters for the initialize request."""

    protocolVersion: str | int
    """The latest version of the Model Context Protocol that the client supports."""
    capabilities: ClientCapabilities
    clientInfo: Implementation
    model_config = ConfigDict(extra="allow")

# cdg:定义初始化请求的模型。
class InitializeRequest(Request[InitializeRequestParams, Literal["initialize"]]):
    """
    This request is sent from the client to the server when it first connects, asking it
    to begin initialization.
    """

    method: Literal["initialize"]
    params: InitializeRequestParams

# cdg:定义初始化结果的模型。
class InitializeResult(Result):
    """After receiving an initialize request from the client, the server sends this."""

    protocolVersion: str | int
    """The version of the Model Context Protocol that the server wants to use."""
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None = None
    """Instructions describing how to use the server and its features."""

# cdg:定义初始化通知的模型。
class InitializedNotification(Notification[NotificationParams | None, Literal["notifications/initialized"]]):
    """
    This notification is sent from the client to the server after initialization has
    finished.
    """

    method: Literal["notifications/initialized"]
    params: NotificationParams | None = None

# cdg:定义Ping请求的模型。
class PingRequest(Request[RequestParams | None, Literal["ping"]]):
    """
    A ping, issued by either the server or the client, to check that the other party is
    still alive.
    """

    method: Literal["ping"]
    params: RequestParams | None = None

# cdg:定义进度通知参数的模型。
class ProgressNotificationParams(NotificationParams):
    """Parameters for progress notifications."""

    progressToken: ProgressToken
    """
    The progress token which was given in the initial request, used to associate this
    notification with the request that is proceeding.
    """
    progress: float
    """
    The progress thus far. This should increase every time progress is made, even if the
    total is unknown.
    """
    total: float | None = None
    """Total number of items to process (or total progress required), if known."""
    model_config = ConfigDict(extra="allow")

# cdg:定义进度通知的模型。
class ProgressNotification(Notification[ProgressNotificationParams, Literal["notifications/progress"]]):
    """
    An out-of-band notification used to inform the receiver of a progress update for a
    long-running request.
    """

    method: Literal["notifications/progress"]
    params: ProgressNotificationParams

# cdg:定义资源列表请求的模型。      
class ListResourcesRequest(PaginatedRequest[RequestParams | None, Literal["resources/list"]]):
    """Sent from the client to request a list of resources the server has."""

    method: Literal["resources/list"]
    params: RequestParams | None = None

# cdg:定义注解的模型。
class Annotations(BaseModel):
    audience: list[Role] | None = None
    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义资源的模型。
class Resource(BaseModel):
    """A known resource that the server is capable of reading."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of this resource."""
    name: str
    """A human-readable name for this resource."""
    description: str | None = None
    """A description of what this resource represents."""
    mimeType: str | None = None
    """The MIME type of this resource, if known."""
    size: int | None = None
    """
    The size of the raw resource content, in bytes (i.e., before base64 encoding
    or any tokenization), if known.

    This can be used by Hosts to display file sizes and estimate context window usage.
    """
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义资源模板的模型。
class ResourceTemplate(BaseModel):
    """A template description for resources available on the server."""

    uriTemplate: str
    """
    A URI template (according to RFC 6570) that can be used to construct resource
    URIs.
    """
    name: str
    """A human-readable name for the type of resource this template refers to."""
    description: str | None = None
    """A human-readable description of what this template is for."""
    mimeType: str | None = None
    """
    The MIME type for all resources that match this template. This should only be
    included if all resources matching this template have the same type.
    """
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义资源列表结果的模型。
class ListResourcesResult(PaginatedResult):
    """The server's response to a resources/list request from the client."""

    resources: list[Resource]

# cdg:定义资源模板列表请求的模型。
class ListResourceTemplatesRequest(PaginatedRequest[RequestParams | None, Literal["resources/templates/list"]]):
    """Sent from the client to request a list of resource templates the server has."""

    method: Literal["resources/templates/list"]
    params: RequestParams | None = None

# cdg:定义资源模板列表结果的模型。
class ListResourceTemplatesResult(PaginatedResult):
    """The server's response to a resources/templates/list request from the client."""

    resourceTemplates: list[ResourceTemplate]

# cdg:定义读取资源请求参数的模型。
class ReadResourceRequestParams(RequestParams):
    """Parameters for reading a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource to read. The URI can use any protocol; it is up to the
    server how to interpret it.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义读取资源请求的模型。
class ReadResourceRequest(Request[ReadResourceRequestParams, Literal["resources/read"]]):
    """Sent from the client to the server, to read a specific resource URI."""

    method: Literal["resources/read"]
    params: ReadResourceRequestParams

# cdg:定义资源内容的模型。
class ResourceContents(BaseModel):
    """The contents of a specific resource or sub-resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of this resource."""
    mimeType: str | None = None
    """The MIME type of this resource, if known."""
    model_config = ConfigDict(extra="allow")

# cdg:定义文本资源内容的模型。
class TextResourceContents(ResourceContents):
    """Text contents of a resource."""

    text: str
    """
    The text of the item. This must only be set if the item can actually be represented
    as text (not binary data).
    """

# cdg:定义二进制资源内容的模型。
class BlobResourceContents(ResourceContents):
    """Binary contents of a resource."""

    blob: str
    """A base64-encoded string representing the binary data of the item."""

# cdg:定义读取资源结果的模型。
class ReadResourceResult(Result):
    """The server's response to a resources/read request from the client."""

    contents: list[TextResourceContents | BlobResourceContents]

# cdg:定义资源列表改变通知的模型。
class ResourceListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/resources/list_changed"]]
):
    """
    An optional notification from the server to the client, informing it that the list
    of resources it can read from has changed.
    """

    method: Literal["notifications/resources/list_changed"]
    params: NotificationParams | None = None

# cdg:定义订阅请求参数的模型。
class SubscribeRequestParams(RequestParams):
    """Parameters for subscribing to a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource to subscribe to. The URI can use any protocol; it is up to
    the server how to interpret it.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义订阅请求的模型。
class SubscribeRequest(Request[SubscribeRequestParams, Literal["resources/subscribe"]]):
    """
    Sent from the client to request resources/updated notifications from the server
    whenever a particular resource changes.
    """

    method: Literal["resources/subscribe"]
    params: SubscribeRequestParams

# cdg:定义取消订阅请求参数的模型。
class UnsubscribeRequestParams(RequestParams):
    """Parameters for unsubscribing from a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of the resource to unsubscribe from."""
    model_config = ConfigDict(extra="allow")

# cdg:定义取消订阅请求的模型。
class UnsubscribeRequest(Request[UnsubscribeRequestParams, Literal["resources/unsubscribe"]]):
    """
    Sent from the client to request cancellation of resources/updated notifications from
    the server.
    """

    method: Literal["resources/unsubscribe"]
    params: UnsubscribeRequestParams

# cdg:定义资源更新通知参数的模型。  
class ResourceUpdatedNotificationParams(NotificationParams):
    """Parameters for resource update notifications."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource that has been updated. This might be a sub-resource of the
    one that the client actually subscribed to.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义资源更新通知的模型。
class ResourceUpdatedNotification(
    Notification[ResourceUpdatedNotificationParams, Literal["notifications/resources/updated"]]
):
    """
    A notification from the server to the client, informing it that a resource has
    changed and may need to be read again.
    """

    method: Literal["notifications/resources/updated"]
    params: ResourceUpdatedNotificationParams

# cdg:定义提示列表请求的模型。
class ListPromptsRequest(PaginatedRequest[RequestParams | None, Literal["prompts/list"]]):
    """Sent from the client to request a list of prompts and prompt templates."""

    method: Literal["prompts/list"]
    params: RequestParams | None = None

# cdg:定义提示参数的模型。
class PromptArgument(BaseModel):
    """An argument for a prompt template."""

    name: str
    """The name of the argument."""
    description: str | None = None
    """A human-readable description of the argument."""
    required: bool | None = None
    """Whether this argument must be provided."""
    model_config = ConfigDict(extra="allow")

# cdg:定义提示的模型。
class Prompt(BaseModel):
    """A prompt or prompt template that the server offers."""

    name: str
    """The name of the prompt or prompt template."""
    description: str | None = None
    """An optional description of what this prompt provides."""
    arguments: list[PromptArgument] | None = None
    """A list of arguments to use for templating the prompt."""
    model_config = ConfigDict(extra="allow")

# cdg:定义提示列表结果的模型。
class ListPromptsResult(PaginatedResult):
    """The server's response to a prompts/list request from the client."""

    prompts: list[Prompt]

# cdg:定义获取提示请求参数的模型。
class GetPromptRequestParams(RequestParams):
    """Parameters for getting a prompt."""

    name: str
    """The name of the prompt or prompt template."""
    arguments: dict[str, str] | None = None
    """Arguments to use for templating the prompt."""
    model_config = ConfigDict(extra="allow")

# cdg:定义获取提示请求的模型。
class GetPromptRequest(Request[GetPromptRequestParams, Literal["prompts/get"]]):
    """Used by the client to get a prompt provided by the server."""

    method: Literal["prompts/get"]
    params: GetPromptRequestParams

# cdg:定义文本内容的模型。
class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"]
    text: str
    """The text content of the message."""
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义图片内容的模型。
class ImageContent(BaseModel):
    """Image content for a message."""

    type: Literal["image"]
    data: str
    """The base64-encoded image data."""
    mimeType: str
    """
    The MIME type of the image. Different providers may support different
    image types.
    """
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义采样消息的模型。
class SamplingMessage(BaseModel):
    """Describes a message issued to or received from an LLM API."""

    role: Role
    content: TextContent | ImageContent
    model_config = ConfigDict(extra="allow")

# cdg:定义嵌入资源的模型。  
class EmbeddedResource(BaseModel):
    """
    The contents of a resource, embedded into a prompt or tool call result.

    It is up to the client how best to render embedded resources for the benefit
    of the LLM and/or the user.
    """

    type: Literal["resource"]
    resource: TextResourceContents | BlobResourceContents
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义提示消息的模型。
class PromptMessage(BaseModel):
    """Describes a message returned as part of a prompt."""

    role: Role
    content: TextContent | ImageContent | EmbeddedResource
    model_config = ConfigDict(extra="allow")

# cdg:定义获取提示结果的模型。
class GetPromptResult(Result):
    """The server's response to a prompts/get request from the client."""

    description: str | None = None
    """An optional description for the prompt."""
    messages: list[PromptMessage]

# cdg:定义提示列表改变通知的模型。
class PromptListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/prompts/list_changed"]]
):
    """
    An optional notification from the server to the client, informing it that the list
    of prompts it offers has changed.
    """

    method: Literal["notifications/prompts/list_changed"]
    params: NotificationParams | None = None

# cdg:定义工具列表请求的模型。
class ListToolsRequest(PaginatedRequest[RequestParams | None, Literal["tools/list"]]):
    """Sent from the client to request a list of tools the server has."""

    method: Literal["tools/list"]
    params: RequestParams | None = None

# cdg:定义工具注解的模型。
class ToolAnnotations(BaseModel):
    """
    Additional properties describing a Tool to clients.

    NOTE: all properties in ToolAnnotations are **hints**.
    They are not guaranteed to provide a faithful description of
    tool behavior (including descriptive properties like `title`).

    Clients should never make tool use decisions based on ToolAnnotations
    received from untrusted servers.
    """

    title: str | None = None
    """A human-readable title for the tool."""

    readOnlyHint: bool | None = None
    """
    If true, the tool does not modify its environment.
    Default: false
    """

    destructiveHint: bool | None = None
    """
    If true, the tool may perform destructive updates to its environment.
    If false, the tool performs only additive updates.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: true
    """

    idempotentHint: bool | None = None
    """
    If true, calling the tool repeatedly with the same arguments
    will have no additional effect on the its environment.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: false
    """

    openWorldHint: bool | None = None
    """
    If true, this tool may interact with an "open world" of external
    entities. If false, the tool's domain of interaction is closed.
    For example, the world of a web search tool is open, whereas that
    of a memory tool is not.
    Default: true
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义工具的模型。
class Tool(BaseModel):
    """Definition for a tool the client can call."""

    name: str
    """The name of the tool."""
    description: str | None = None
    """A human-readable description of the tool."""
    inputSchema: dict[str, Any]
    """A JSON Schema object defining the expected parameters for the tool."""
    annotations: ToolAnnotations | None = None
    """Optional additional tool information."""
    model_config = ConfigDict(extra="allow")

# cdg:定义工具列表结果的模型。
class ListToolsResult(PaginatedResult):
    """The server's response to a tools/list request from the client."""

    tools: list[Tool]

# cdg:定义调用工具请求参数的模型。
class CallToolRequestParams(RequestParams):
    """Parameters for calling a tool."""

    name: str
    arguments: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")

# cdg:定义调用工具请求的模型。
class CallToolRequest(Request[CallToolRequestParams, Literal["tools/call"]]):
    """Used by the client to invoke a tool provided by the server."""

    method: Literal["tools/call"]
    params: CallToolRequestParams

# cdg:定义调用工具结果的模型。
class CallToolResult(Result):
    """The server's response to a tool call."""

    content: list[TextContent | ImageContent | EmbeddedResource]
    isError: bool = False

# cdg:定义工具列表改变通知的模型。
class ToolListChangedNotification(Notification[NotificationParams | None, Literal["notifications/tools/list_changed"]]):
    """
    An optional notification from the server to the client, informing it that the list
    of tools it offers has changed.
    """

    method: Literal["notifications/tools/list_changed"]
    params: NotificationParams | None = None

# cdg:日志级别
LoggingLevel = Literal["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]

# cdg:定义设置日志级别请求参数的模型。
class SetLevelRequestParams(RequestParams):
    """Parameters for setting the logging level."""

    level: LoggingLevel
    """The level of logging that the client wants to receive from the server."""
    model_config = ConfigDict(extra="allow")

# cdg:定义设置日志级别请求的模型。
class SetLevelRequest(Request[SetLevelRequestParams, Literal["logging/setLevel"]]):
    """A request from the client to the server, to enable or adjust logging."""

    method: Literal["logging/setLevel"]
    params: SetLevelRequestParams

# cdg:定义日志消息通知参数的模型。
class LoggingMessageNotificationParams(NotificationParams):
    """Parameters for logging message notifications."""

    level: LoggingLevel
    """The severity of this log message."""
    logger: str | None = None
    """An optional name of the logger issuing this message."""
    data: Any
    """
    The data to be logged, such as a string message or an object. Any JSON serializable
    type is allowed here.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义日志消息通知的模型。
class LoggingMessageNotification(Notification[LoggingMessageNotificationParams, Literal["notifications/message"]]):
    """Notification of a log message passed from server to client."""

    method: Literal["notifications/message"]
    params: LoggingMessageNotificationParams


IncludeContext = Literal["none", "thisServer", "allServers"]

# cdg:定义模型提示的模型。
class ModelHint(BaseModel):
    """Hints to use for model selection."""

    name: str | None = None
    """A hint for a model name."""

    model_config = ConfigDict(extra="allow")

# cdg:定义模型偏好的模型。
class ModelPreferences(BaseModel):
    """
    The server's preferences for model selection, requested by the client during
    sampling.

    Because LLMs can vary along multiple dimensions, choosing the "best" model is
    rarely straightforward.  Different models excel in different areas—some are
    faster but less capable, others are more capable but more expensive, and so
    on. This interface allows servers to express their priorities across multiple
    dimensions to help clients make an appropriate selection for their use case.

    These preferences are always advisory. The client MAY ignore them. It is also
    up to the client to decide how to interpret these preferences and how to
    balance them against other considerations.
    """

    hints: list[ModelHint] | None = None
    """
    Optional hints to use for model selection.

    If multiple hints are specified, the client MUST evaluate them in order
    (such that the first match is taken).

    The client SHOULD prioritize these hints over the numeric priorities, but
    MAY still use the priorities to select from ambiguous matches.
    """

    costPriority: float | None = None
    """
    How much to prioritize cost when selecting a model. A value of 0 means cost
    is not important, while a value of 1 means cost is the most important
    factor.
    """

    speedPriority: float | None = None
    """
    How much to prioritize sampling speed (latency) when selecting a model. A
    value of 0 means speed is not important, while a value of 1 means speed is
    the most important factor.
    """

    intelligencePriority: float | None = None
    """
    How much to prioritize intelligence and capabilities when selecting a
    model. A value of 0 means intelligence is not important, while a value of 1
    means intelligence is the most important factor.
    """

    model_config = ConfigDict(extra="allow")

# cdg:定义创建消息请求参数的模型。
class CreateMessageRequestParams(RequestParams):
    """Parameters for creating a message."""

    messages: list[SamplingMessage]
    modelPreferences: ModelPreferences | None = None
    """
    The server's preferences for which model to select. The client MAY ignore
    these preferences.
    """
    systemPrompt: str | None = None
    """An optional system prompt the server wants to use for sampling."""
    includeContext: IncludeContext | None = None
    """
    A request to include context from one or more MCP servers (including the caller), to
    be attached to the prompt.
    """
    temperature: float | None = None
    maxTokens: int
    """The maximum number of tokens to sample, as requested by the server."""
    stopSequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    """Optional metadata to pass through to the LLM provider."""
    model_config = ConfigDict(extra="allow")

# cdg:定义创建消息请求的模型。
class CreateMessageRequest(Request[CreateMessageRequestParams, Literal["sampling/createMessage"]]):
    """A request from the server to sample an LLM via the client."""

    method: Literal["sampling/createMessage"]
    params: CreateMessageRequestParams

# cdg:停止原因
StopReason = Literal["endTurn", "stopSequence", "maxTokens"] | str

# cdg:定义创建消息结果的模型。
class CreateMessageResult(Result):
    """The client's response to a sampling/create_message request from the server."""

    role: Role
    content: TextContent | ImageContent
    model: str
    """The name of the model that generated the message."""
    stopReason: StopReason | None = None
    """The reason why sampling stopped, if known."""

# cdg:定义资源引用的模型。
class ResourceReference(BaseModel):
    """A reference to a resource or resource template definition."""

    type: Literal["ref/resource"]
    uri: str
    """The URI or URI template of the resource."""
    model_config = ConfigDict(extra="allow")

# cdg:定义提示引用的模型。
class PromptReference(BaseModel):
    """Identifies a prompt."""

    type: Literal["ref/prompt"]
    name: str
    """The name of the prompt or prompt template"""
    model_config = ConfigDict(extra="allow")

# cdg:定义Completion参数的模型。
class CompletionArgument(BaseModel):
    """The argument's information for completion requests."""

    name: str
    """The name of the argument"""
    value: str
    """The value of the argument to use for completion matching."""
    model_config = ConfigDict(extra="allow")

# cdg:定义Completion请求参数的模型。
class CompleteRequestParams(RequestParams):
    """Parameters for completion requests."""

    ref: ResourceReference | PromptReference
    argument: CompletionArgument
    model_config = ConfigDict(extra="allow")

# cdg:定义Completion请求的模型。
class CompleteRequest(Request[CompleteRequestParams, Literal["completion/complete"]]):
    """A request from the client to the server, to ask for completion options."""

    method: Literal["completion/complete"]
    params: CompleteRequestParams

# cdg:定义Completion的模型。
class Completion(BaseModel):
    """Completion information."""

    values: list[str]
    """An array of completion values. Must not exceed 100 items."""
    total: int | None = None
    """
    The total number of completion options available. This can exceed the number of
    values actually sent in the response.
    """
    hasMore: bool | None = None
    """
    Indicates whether there are additional completion options beyond those provided in
    the current response, even if the exact total is unknown.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义Completion结果的模型。
class CompleteResult(Result):
    """The server's response to a completion/complete request"""

    completion: Completion

# cdg:定义根列表请求的模型。
class ListRootsRequest(Request[RequestParams | None, Literal["roots/list"]]):
    """
    Sent from the server to request a list of root URIs from the client. Roots allow
    servers to ask for specific directories or files to operate on. A common example
    for roots is providing a set of repositories or directories a server should operate
    on.

    This request is typically used when the server needs to understand the file system
    structure or access specific locations that the client has permission to read from.
    """

    method: Literal["roots/list"]
    params: RequestParams | None = None

# cdg:定义根的模型。
class Root(BaseModel):
    """Represents a root directory or file that the server can operate on."""

    uri: FileUrl
    """
    The URI identifying the root. This *must* start with file:// for now.
    This restriction may be relaxed in future versions of the protocol to allow
    other URI schemes.
    """
    name: str | None = None
    """
    An optional name for the root. This can be used to provide a human-readable
    identifier for the root, which may be useful for display purposes or for
    referencing the root in other parts of the application.
    """
    model_config = ConfigDict(extra="allow")

# cdg:定义根列表结果的模型。
class ListRootsResult(Result):
    """
    The client's response to a roots/list request from the server.
    This result contains an array of Root objects, each representing a root directory
    or file that the server can operate on.
    """

    roots: list[Root]

# cdg:定义根列表改变通知的模型。
class RootsListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/roots/list_changed"]]
):
    """
    A notification from the client to the server, informing it that the list of
    roots has changed.

    This notification should be sent whenever the client adds, removes, or
    modifies any root. The server should then request an updated list of roots
    using the ListRootsRequest.
    """

    method: Literal["notifications/roots/list_changed"]
    params: NotificationParams | None = None

# cdg:定义取消通知参数的模型。
class CancelledNotificationParams(NotificationParams):
    """Parameters for cancellation notifications."""

    requestId: RequestId
    """The ID of the request to cancel."""
    reason: str | None = None
    """An optional string describing the reason for the cancellation."""
    model_config = ConfigDict(extra="allow")

# cdg:定义取消通知的模型。
class CancelledNotification(Notification[CancelledNotificationParams, Literal["notifications/cancelled"]]):
    """
    This notification can be sent by either side to indicate that it is canceling a
    previously-issued request.
    """

    method: Literal["notifications/cancelled"]
    params: CancelledNotificationParams

# cdg:定义客户端请求的模型,包括ping、初始化、完成、设置日志级别、获取提示、列表提示、列表资源、列表资源模板、读取资源、订阅、取消订阅、调用工具、列表工具
class ClientRequest(
    RootModel[
        PingRequest
        | InitializeRequest
        | CompleteRequest
        | SetLevelRequest
        | GetPromptRequest
        | ListPromptsRequest
        | ListResourcesRequest
        | ListResourceTemplatesRequest
        | ReadResourceRequest
        | SubscribeRequest
        | UnsubscribeRequest
        | CallToolRequest
        | ListToolsRequest
    ]
):
    pass

# cdg:定义客户端通知的模型,可以是取消、进度、初始化、根列表改变
class ClientNotification(
    RootModel[CancelledNotification | ProgressNotification | InitializedNotification | RootsListChangedNotification]
):
    pass

# cdg:定义客户端结果的模型,可以是空、创建消息、列表根
class ClientResult(RootModel[EmptyResult | CreateMessageResult | ListRootsResult]):
    pass

# cdg:定义服务器请求的模型,可以是ping、创建消息、列表根
class ServerRequest(RootModel[PingRequest | CreateMessageRequest | ListRootsRequest]):
    pass

# cdg:定义服务器通知的模型,可以是取消、进度、日志消息、资源更新、资源列表改变、工具列表改变、提示列表改变
class ServerNotification(
    RootModel[
        CancelledNotification
        | ProgressNotification
        | LoggingMessageNotification
        | ResourceUpdatedNotification
        | ResourceListChangedNotification
        | ToolListChangedNotification
        | PromptListChangedNotification
    ]
):
    pass

# cdg:定义服务器结果的模型,可以是空、初始化、完成、获取提示、列表提示、列表资源、列表资源模板、读取资源、调用工具、列表工具
class ServerResult(
    RootModel[
        EmptyResult
        | InitializeResult
        | CompleteResult
        | GetPromptResult
        | ListPromptsResult
        | ListResourcesResult
        | ListResourceTemplatesResult
        | ReadResourceResult
        | CallToolResult
        | ListToolsResult
    ]
):
    pass


ResumptionToken = str

ResumptionTokenUpdateCallback = Callable[[ResumptionToken], None]

# cdg:定义客户端消息元数据的模型,包含重续令牌、重续令牌更新回调，@dataclass是python的装饰器，用于将类转换为数据类
@dataclass
class ClientMessageMetadata:
    """Metadata specific to client messages."""

    resumption_token: ResumptionToken | None = None
    on_resumption_token_update: Callable[[ResumptionToken], None] | None = None

# cdg:定义服务器消息元数据的模型,包含相关请求ID、请求上下文
@dataclass
class ServerMessageMetadata:
    """Metadata specific to server messages."""

    related_request_id: RequestId | None = None
    request_context: object | None = None

# cdg:定义消息元数据的模型,可以是客户端消息元数据、服务器消息元数据、空
MessageMetadata = ClientMessageMetadata | ServerMessageMetadata | None

# cdg:定义会话消息的模型,包含消息、元数据
@dataclass
class SessionMessage:
    """A message with specific metadata for transport-specific features."""

    message: JSONRPCMessage
    metadata: MessageMetadata = None

# cdg:定义OAuth客户端元数据的模型,包含客户端名称、重定向URI、授权类型、响应类型、令牌端点认证方法、客户端URI、范围
class OAuthClientMetadata(BaseModel):
    client_name: str
    redirect_uris: list[str]
    grant_types: Optional[list[str]] = None
    response_types: Optional[list[str]] = None
    token_endpoint_auth_method: Optional[str] = None
    client_uri: Optional[str] = None
    scope: Optional[str] = None

# cdg:定义OAuth客户端信息的模型,包含客户端ID、客户端密钥
class OAuthClientInformation(BaseModel):
    client_id: str
    client_secret: Optional[str] = None

# cdg:定义OAuth客户端信息的完整模型,包含客户端名称、重定向URI、范围、授权类型、响应类型、令牌端点认证方法
class OAuthClientInformationFull(OAuthClientInformation):
    client_name: str | None = None
    redirect_uris: list[str]
    scope: Optional[str] = None
    grant_types: Optional[list[str]] = None
    response_types: Optional[list[str]] = None
    token_endpoint_auth_method: Optional[str] = None

# cdg:定义OAuth令牌的模型,包含访问令牌、令牌类型、过期时间、刷新令牌、范围
class OAuthTokens(BaseModel):
    access_token: str
    token_type: str
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

# cdg:定义OAuth元数据的模型,包含授权端点、令牌端点、注册端点、响应类型支持、授权类型支持、代码挑战方法支持
class OAuthMetadata(BaseModel):
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: Optional[str] = None
    response_types_supported: list[str]
    grant_types_supported: Optional[list[str]] = None
    code_challenge_methods_supported: Optional[list[str]] = None
