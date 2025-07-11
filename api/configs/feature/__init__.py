from typing import Annotated, Literal, Optional

from pydantic import (
    AliasChoices,
    Field,
    HttpUrl,
    NegativeInt,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
)
from pydantic_settings import BaseSettings

from configs.feature.hosted_service import HostedServiceConfig

# cdg: 安全相关配置信息
class SecurityConfig(BaseSettings):
    """
    Security-related configurations for the application
    """
    # cdg: 安全密钥
    SECRET_KEY: str = Field(
        description="Secret key for secure session cookie signing."
        "Make sure you are changing this key for your deployment with a strong key."
        "Generate a strong key using `openssl rand -base64 42` or set via the `SECRET_KEY` environment variable.",
        default="",
    )
    # cdg: 重置密码令牌过期时间
    RESET_PASSWORD_TOKEN_EXPIRY_MINUTES: PositiveInt = Field(
        description="Duration in minutes for which a password reset token remains valid",
        default=5,
    )
    # cdg: 是否禁用登录
    LOGIN_DISABLED: bool = Field(
        description="Whether to disable login checks",
        default=False,
    )
    # cdg: 是否启用管理员API密钥
    ADMIN_API_KEY_ENABLE: bool = Field(
        description="Whether to enable admin api key for authentication",
        default=False,
    )
    # cdg: 管理员API密钥
    ADMIN_API_KEY: Optional[str] = Field(
        description="admin api key for authentication",
        default=None,
    )

# cdg: 应用执行配置信息
class AppExecutionConfig(BaseSettings):
    """
    Configuration parameters for application execution
    """
    # cdg: 应用最大执行时间
    APP_MAX_EXECUTION_TIME: PositiveInt = Field(
        description="Maximum allowed execution time for the application in seconds",
        default=1200,
    )
    # cdg: 应用最大并发请求数
    APP_MAX_ACTIVE_REQUESTS: NonNegativeInt = Field(
        description="Maximum number of concurrent active requests per app (0 for unlimited)",
        default=0,
    )
# cdg: 代码执行沙盒配置信息
class CodeExecutionSandboxConfig(BaseSettings):
    """
    Configuration for the code execution sandbox environment
    """
    # cdg: 代码执行沙盒URL
    CODE_EXECUTION_ENDPOINT: HttpUrl = Field(
        description="URL endpoint for the code execution service",
        default="http://sandbox:8194",
    )
    # cdg: 代码执行沙盒API密钥
    CODE_EXECUTION_API_KEY: str = Field(
        description="API key for accessing the code execution service",
        default="dify-sandbox",
    )
    # cdg: 代码执行沙盒连接超时时间
    CODE_EXECUTION_CONNECT_TIMEOUT: Optional[float] = Field(
        description="Connection timeout in seconds for code execution requests",
        default=10.0,
    )
    # cdg: 代码执行沙盒读取超时时间
    CODE_EXECUTION_READ_TIMEOUT: Optional[float] = Field(
        description="Read timeout in seconds for code execution requests",
        default=60.0,
    )

    CODE_EXECUTION_WRITE_TIMEOUT: Optional[float] = Field(
        description="Write timeout in seconds for code execution request",
        default=10.0,
    )
    # cdg: 代码执行沙盒最大数字
    CODE_MAX_NUMBER: PositiveInt = Field(
        description="Maximum allowed numeric value in code execution",
        default=9223372036854775807,
    )
    # cdg: 代码执行沙盒最小数字
    CODE_MIN_NUMBER: NegativeInt = Field(
        description="Minimum allowed numeric value in code execution",
        default=-9223372036854775807,
    )
    # cdg: 代码执行沙盒最大深度
    CODE_MAX_DEPTH: PositiveInt = Field(
        description="Maximum allowed depth for nested structures in code execution",
        default=5,
    )
    # cdg: 代码执行沙盒最大精度
    CODE_MAX_PRECISION: PositiveInt = Field(
        description="Maximum number of decimal places for floating-point numbers in code execution",
        default=20,
    )
    # cdg: 代码执行沙盒最大字符串长度
    CODE_MAX_STRING_LENGTH: PositiveInt = Field(
        description="Maximum allowed length for strings in code execution",
        default=80000,
    )
    # cdg: 代码执行沙盒最大字符串数组长度   
    CODE_MAX_STRING_ARRAY_LENGTH: PositiveInt = Field(
        description="Maximum allowed length for string arrays in code execution",
        default=30,
    )
    # cdg: 代码执行沙盒最大对象数组长度
    CODE_MAX_OBJECT_ARRAY_LENGTH: PositiveInt = Field(
        description="Maximum allowed length for object arrays in code execution",
        default=30,
    )
    # cdg: 代码执行沙盒最大数字数组长度
    CODE_MAX_NUMBER_ARRAY_LENGTH: PositiveInt = Field(
        description="Maximum allowed length for numeric arrays in code execution",
        default=1000,
    )
# cdg: 应用端点配置信息
class EndpointConfig(BaseSettings):
    """
    Configuration for various application endpoints and URLs
    """
    # cdg: 控制台API URL
    CONSOLE_API_URL: str = Field(
        description="Base URL for the console API,"
        "used for login authentication callback or notion integration callbacks",
        default="",
    )
    # cdg: 控制台Web URL
    CONSOLE_WEB_URL: str = Field(
        description="Base URL for the console web interface," "used for frontend references and CORS configuration",
        default="",
    )
    # cdg: 服务API URL
    SERVICE_API_URL: str = Field(
        description="Base URL for the service API, displayed to users for API access",
        default="",
    )
    # cdg: 应用Web URL
    APP_WEB_URL: str = Field(
        description="Base URL for the web application, used for frontend references",
        default="",
    )

# cdg: 文件访问配置信息
class FileAccessConfig(BaseSettings):
    """
    Configuration for file access and handling
    """
    # cdg: 文件URL
    FILES_URL: str = Field(
        description="Base URL for file preview or download,"
        " used for frontend display and multi-model inputs"
        "Url is signed and has expiration time.",
        validation_alias=AliasChoices("FILES_URL", "CONSOLE_API_URL"),
        alias_priority=1,
        default="",
    )
    # cdg: 文件访问超时时间
    FILES_ACCESS_TIMEOUT: int = Field(
        description="Expiration time in seconds for file access URLs",
        default=300,
    )


# cdg: 文件上传配置信息 
class FileUploadConfig(BaseSettings):
    """
    Configuration for file upload limitations
    """
    # cdg: 文件上传大小限制
    UPLOAD_FILE_SIZE_LIMIT: NonNegativeInt = Field(
        description="Maximum allowed file size for uploads in megabytes",
        default=15,
    )
    # cdg: 文件上传批量限制
    UPLOAD_FILE_BATCH_LIMIT: NonNegativeInt = Field(
        description="Maximum number of files allowed in a single upload batch",
        default=5,
    )
    # cdg: 图片文件上传大小限制
    UPLOAD_IMAGE_FILE_SIZE_LIMIT: NonNegativeInt = Field(
        description="Maximum allowed image file size for uploads in megabytes",
        default=10,
    )
    # cdg: 视频文件上传大小限制
    UPLOAD_VIDEO_FILE_SIZE_LIMIT: NonNegativeInt = Field(
        description="video file size limit in Megabytes for uploading files",
        default=100,
    )
    # cdg: 音频文件上传大小限制
    UPLOAD_AUDIO_FILE_SIZE_LIMIT: NonNegativeInt = Field(
        description="audio file size limit in Megabytes for uploading files",
        default=50,
    )
    # cdg: 批量上传限制
    BATCH_UPLOAD_LIMIT: NonNegativeInt = Field(
        description="Maximum number of files allowed in a batch upload operation",
        default=20,
    )
    # cdg: 工作流文件上传限制
    WORKFLOW_FILE_UPLOAD_LIMIT: PositiveInt = Field(
        description="Maximum number of files allowed in a workflow upload operation",
        default=10,
    )
# cdg: HTTP相关配置信息
class HttpConfig(BaseSettings):
    """
    HTTP-related configurations for the application
    """
    # cdg: 是否启用API压缩
    API_COMPRESSION_ENABLED: bool = Field(
        description="Enable or disable gzip compression for HTTP responses",
        default=False,
    )
    # cdg: 内部控制台CORS允许源。用于配置内部控制台的CORS允许源，即允许哪些域名或IP地址访问内部控制台。
    inner_CONSOLE_CORS_ALLOW_ORIGINS: str = Field(
        description="Comma-separated list of allowed origins for CORS in the console",
        validation_alias=AliasChoices("CONSOLE_CORS_ALLOW_ORIGINS", "CONSOLE_WEB_URL"),
        default="",
    )
    # cdg: 内部控制台CORS允许源列表。用于将内部控制台CORS允许源转换为列表。
    # cdg: 内部控制台CORS允许源列表。用于将内部控制台CORS允许源转换为列表。
    @computed_field
    def CONSOLE_CORS_ALLOW_ORIGINS(self) -> list[str]:
        return self.inner_CONSOLE_CORS_ALLOW_ORIGINS.split(",")
    
    # cdg: 内部Web API CORS允许源。用于配置内部Web API的CORS允许源，即允许哪些域名或IP地址访问内部Web API。
    inner_WEB_API_CORS_ALLOW_ORIGINS: str = Field(
        description="",
        validation_alias=AliasChoices("WEB_API_CORS_ALLOW_ORIGINS"),
        default="*",
    )
    # cdg: 内部Web API CORS允许源列表。用于将内部Web API CORS允许源转换为列表。
    @computed_field
    def WEB_API_CORS_ALLOW_ORIGINS(self) -> list[str]:
        return self.inner_WEB_API_CORS_ALLOW_ORIGINS.split(",")

    # cdg: HTTP请求节点最大连接超时时间。用于配置HTTP请求的最大连接超时时间，即HTTP请求在连接建立后，等待响应的最大时间。
    HTTP_REQUEST_MAX_CONNECT_TIMEOUT: Annotated[
        PositiveInt, Field(ge=10, description="Maximum connection timeout in seconds for HTTP requests")
    ] = 10

    # cdg: HTTP请求最大读取超时时间。用于配置HTTP请求的最大读取超时时间，即HTTP请求在读取响应数据时，等待响应的最大时间。
    HTTP_REQUEST_MAX_READ_TIMEOUT: Annotated[
        PositiveInt, Field(ge=60, description="Maximum read timeout in seconds for HTTP requests")
    ] = 60

    # cdg: HTTP请求最大写入超时时间。用于配置HTTP请求的最大写入超时时间，即HTTP请求在写入请求数据时，等待响应的最大时间。
    HTTP_REQUEST_MAX_WRITE_TIMEOUT: Annotated[
        PositiveInt, Field(ge=10, description="Maximum write timeout in seconds for HTTP requests")
    ] = 20

    # cdg: HTTP请求节点最大二进制数据大小。用于配置HTTP请求节点可以接收的最大二进制数据大小，即HTTP请求节点可以接收的最大二进制数据大小。
    HTTP_REQUEST_NODE_MAX_BINARY_SIZE: PositiveInt = Field(
        description="Maximum allowed size in bytes for binary data in HTTP requests",
        default=10 * 1024 * 1024,
    )
    # cdg: HTTP请求节点最大文本数据大小。用于配置HTTP请求节点可以接收的最大文本数据大小，即HTTP请求节点可以接收的最大文本数据大小。
    HTTP_REQUEST_NODE_MAX_TEXT_SIZE: PositiveInt = Field(
        description="Maximum allowed size in bytes for text data in HTTP requests",
        default=1 * 1024 * 1024,
    )
    # cdg: HTTP请求节点最大重试次数。用于配置HTTP请求节点可以接收的最大重试次数，即HTTP请求节点可以接收的最大重试次数。
    SSRF_DEFAULT_MAX_RETRIES: PositiveInt = Field(
        description="Maximum number of retries for network requests (SSRF)",
        default=3,
    )
    
    # cdg: SSRF代理URL。用于配置SSRF代理URL，即SSRF代理URL。
    SSRF_PROXY_ALL_URL: Optional[str] = Field(
        description="Proxy URL for HTTP or HTTPS requests to prevent Server-Side Request Forgery (SSRF)",
        default=None,
    )
    # cdg: SSRF代理HTTP URL。用于配置SSRF代理HTTP URL，即SSRF代理HTTP URL。
    SSRF_PROXY_HTTP_URL: Optional[str] = Field(
        description="Proxy URL for HTTP requests to prevent Server-Side Request Forgery (SSRF)",
        default=None,
    )

    # cdg: SSRF的原理：SSRF（Server-Side Request Forgery）是一种攻击方式，攻击者通过构造恶意的URL，诱导服务器执行恶意请求。
    # cdg: SSRF代理HTTPS URL。用于配置SSRF代理HTTPS URL，即SSRF代理HTTPS URL。
    SSRF_PROXY_HTTPS_URL: Optional[str] = Field(
        description="Proxy URL for HTTPS requests to prevent Server-Side Request Forgery (SSRF)",
        default=None,
    )
    # cdg: SSRF默认超时时间。用于配置SSRF默认超时时间，即SSRF默认超时时间。
    SSRF_DEFAULT_TIME_OUT: PositiveFloat = Field(
        description="The default timeout period used for network requests (SSRF)",
        default=5,
    )
    # cdg: SSRF默认连接超时时间。用于配置SSRF默认连接超时时间，即SSRF默认连接超时时间。
    SSRF_DEFAULT_CONNECT_TIME_OUT: PositiveFloat = Field(
        description="The default connect timeout period used for network requests (SSRF)",
        default=5,
    )
    # cdg: SSRF默认读取超时时间。用于配置SSRF默认读取超时时间，即SSRF默认读取超时时间。
    SSRF_DEFAULT_READ_TIME_OUT: PositiveFloat = Field(
        description="The default read timeout period used for network requests (SSRF)",
        default=5,
    )
    # cdg: SSRF默认写入超时时间。用于配置SSRF默认写入超时时间，即SSRF默认写入超时时间。
    SSRF_DEFAULT_WRITE_TIME_OUT: PositiveFloat = Field(
        description="The default write timeout period used for network requests (SSRF)",
        default=5,
    )
    # cdg: 是否启用X-Forwarded-For Proxy Fix中间件。用于配置是否启用X-Forwarded-For Proxy Fix中间件，即是否启用X-Forwarded-For Proxy Fix中间件。
    RESPECT_XFORWARD_HEADERS_ENABLED: bool = Field(
        description="Enable or disable the X-Forwarded-For Proxy Fix middleware from Werkzeug"
        " to respect X-* headers to redirect clients",
        default=False,
    )

# cdg: 内部API配置信息
class InnerAPIConfig(BaseSettings):
    """
    Configuration for internal API functionality
    """
    # cdg: 是否启用内部API。用于配置是否启用内部API，即是否启用内部API。
    INNER_API: bool = Field(
        description="Enable or disable the internal API",
        default=False,
    )
    # cdg: 内部API密钥。用于配置内部API密钥，即内部API密钥。
    INNER_API_KEY: Optional[str] = Field(
        description="API key for accessing the internal API",
        default=None,
    )

# cdg: 日志配置信息
class LoggingConfig(BaseSettings):
    """
    Configuration for application logging
    """
    # cdg: 日志级别。用于配置日志级别，即日志级别。
    LOG_LEVEL: str = Field(
        description="Logging level, default to INFO. Set to ERROR for production environments.",
        default="INFO",
    )
    # cdg: 日志文件路径。用于配置日志文件路径，即日志文件路径。
    LOG_FILE: Optional[str] = Field(
        description="File path for log output.",
        default=None,
    )
    # cdg: 日志文件最大大小。用于配置日志文件最大大小，即日志文件最大大小。
    LOG_FILE_MAX_SIZE: PositiveInt = Field(
        description="Maximum file size for file rotation retention, the unit is megabytes (MB)",
        default=20,
    )
    # cdg: 日志文件备份数量。用于配置日志文件备份数量，即日志文件备份数量。
    LOG_FILE_BACKUP_COUNT: PositiveInt = Field(
        description="Maximum file backup count file rotation retention",
        default=5,
    )
    # cdg: 日志格式。用于配置日志格式，即日志格式。
    LOG_FORMAT: str = Field(
        description="Format string for log messages",
        default="%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] [%(filename)s:%(lineno)d] - %(message)s",
    )
    # cdg: 日志日期格式。用于配置日志日期格式，即日志日期格式。
    LOG_DATEFORMAT: Optional[str] = Field(
        description="Date format string for log timestamps",
        default=None,
    )
    # cdg: 日志时区。用于配置日志时区，即日志时区。
    LOG_TZ: Optional[str] = Field(
        description="Timezone for log timestamps (e.g., 'America/New_York')",
        default="UTC",
    )

# cdg: 模型负载均衡配置信息 
class ModelLoadBalanceConfig(BaseSettings):
    """
    Configuration for model load balancing
    """
    # cdg: 是否启用模型负载均衡。用于配置是否启用模型负载均衡，即是否启用模型负载均衡。
    MODEL_LB_ENABLED: bool = Field(
        description="Enable or disable load balancing for models",
        default=False,
    )

# cdg: 平台计费配置信息
class BillingConfig(BaseSettings):
    """
    Configuration for platform billing features
    """
    # cdg: 是否启用平台计费。用于配置是否启用平台计费，即是否启用平台计费。
    BILLING_ENABLED: bool = Field(
        description="Enable or disable billing functionality",
        default=False,
    )

# cdg: 应用更新检查配置信息
class UpdateConfig(BaseSettings):
    """
    Configuration for application update checks
    """
    # cdg: 应用更新检查URL。用于配置应用更新检查URL，即应用更新检查URL。
    CHECK_UPDATE_URL: str = Field(
        description="URL to check for application updates",
        default="https://updates.dify.ai",
    )

# cdg: 工作流执行配置信息
class WorkflowConfig(BaseSettings):
    """
    Configuration for workflow execution
    """
    # cdg: 工作流最大执行步骤。用于配置工作流最大执行步骤，即工作流最大执行步骤。
    WORKFLOW_MAX_EXECUTION_STEPS: PositiveInt = Field(
        description="Maximum number of steps allowed in a single workflow execution",
        default=500,
    )
    # cdg: 工作流最大执行时间。用于配置工作流最大执行时间，即工作流最大执行时间。
    WORKFLOW_MAX_EXECUTION_TIME: PositiveInt = Field(
        description="Maximum execution time in seconds for a single workflow",
        default=1200,
    )
    # cdg: 工作流调用最大深度。用于配置工作流调用最大深度，即工作流调用最大深度。
    WORKFLOW_CALL_MAX_DEPTH: PositiveInt = Field(
        description="Maximum allowed depth for nested workflow calls",
        default=5,
    )
    # cdg: 工作流并行深度限制。用于配置工作流并行深度限制，即工作流并行深度限制。
    WORKFLOW_PARALLEL_DEPTH_LIMIT: PositiveInt = Field(
        description="Maximum allowed depth for nested parallel executions",
        default=3,
    )
    # cdg: 变量最大大小。用于配置变量最大大小，即变量最大大小。
    MAX_VARIABLE_SIZE: PositiveInt = Field(
        description="Maximum size in bytes for a single variable in workflows. Default to 200 KB.",
        default=200 * 1024,
    )

# cdg: 工作流节点执行配置信息   
class WorkflowNodeExecutionConfig(BaseSettings):
    """
    Configuration for workflow node execution
    """
    # cdg: 最大提交计数。用于配置最大提交计数，即最大提交计数。
    MAX_SUBMIT_COUNT: PositiveInt = Field(
        description="Maximum number of submitted thread count in a ThreadPool for parallel node execution",
        default=100,
    )

# cdg: 认证和OAuth配置信息
class AuthConfig(BaseSettings):
    """
    Configuration for authentication and OAuth
    """
    # cdg: OAuth重定向路径。用于配置OAuth重定向路径，即OAuth重定向路径。
    OAUTH_REDIRECT_PATH: str = Field(
        description="Redirect path for OAuth authentication callbacks",
        default="/console/api/oauth/authorize",
    )
    # cdg: GitHub OAuth客户端ID。用于配置GitHub OAuth客户端ID，即GitHub OAuth客户端ID。
    GITHUB_CLIENT_ID: Optional[str] = Field(
        description="GitHub OAuth client ID",
        default=None,
    )
    # cdg: GitHub OAuth客户端密钥。用于配置GitHub OAuth客户端密钥，即GitHub OAuth客户端密钥。
    GITHUB_CLIENT_SECRET: Optional[str] = Field(
        description="GitHub OAuth client secret",
        default=None,
    )
    # cdg: Google OAuth客户端ID。用于配置Google OAuth客户端ID，即Google OAuth客户端ID。
    GOOGLE_CLIENT_ID: Optional[str] = Field(
        description="Google OAuth client ID",
        default=None,
    )
    # cdg: Google OAuth客户端密钥。用于配置Google OAuth客户端密钥，即Google OAuth客户端密钥。
    GOOGLE_CLIENT_SECRET: Optional[str] = Field(
        description="Google OAuth client secret",
        default=None,
    )
    # cdg: 访问令牌过期时间。用于配置访问令牌过期时间，即访问令牌过期时间。
    ACCESS_TOKEN_EXPIRE_MINUTES: PositiveInt = Field(
        description="Expiration time for access tokens in minutes",
        default=60,
    )
    # cdg: 刷新令牌过期时间。用于配置刷新令牌过期时间，即刷新令牌过期时间。
    REFRESH_TOKEN_EXPIRE_DAYS: PositiveFloat = Field(
        description="Expiration time for refresh tokens in days",
        default=30,
    )
    # cdg: 登录锁定时间。用于配置登录锁定时间，即登录锁定时间。
    LOGIN_LOCKOUT_DURATION: PositiveInt = Field(
        description="Time (in seconds) a user must wait before retrying login after exceeding the rate limit.",
        default=86400,
    )

# cdg: 内容审核配置信息
class ModerationConfig(BaseSettings):
    """
    Configuration for content moderation
    """
    # cdg: 内容审核缓冲区大小。用于配置内容审核缓冲区大小，即内容审核缓冲区大小。
    MODERATION_BUFFER_SIZE: PositiveInt = Field(
        description="Size of the buffer for content moderation processing",
        default=300,
    )

# cdg: 工具管理配置信息
class ToolConfig(BaseSettings):
    """
    Configuration for tool management
    """
    # cdg: 工具图标缓存最大年龄。用于配置工具图标缓存最大年龄，即工具图标缓存最大年龄。
    TOOL_ICON_CACHE_MAX_AGE: PositiveInt = Field(
        description="Maximum age in seconds for caching tool icons",
        default=3600,
    )

# cdg: 邮件服务配置信息
class MailConfig(BaseSettings):
    """
    Configuration for email services
    """
    # cdg: 邮件服务提供商类型。用于配置邮件服务提供商类型，即邮件服务提供商类型。
    MAIL_TYPE: Optional[str] = Field(
        description="Email service provider type ('smtp' or 'resend'), default to None.",
        default=None,
    )
    # cdg: 默认发件人。用于配置默认发件人，即默认发件人。
    MAIL_DEFAULT_SEND_FROM: Optional[str] = Field(
        description="Default email address to use as the sender",
        default=None,
    )
    # cdg: Resend API密钥。用于配置Resend API密钥，即Resend API密钥。
    RESEND_API_KEY: Optional[str] = Field(
        description="API key for Resend email service",
        default=None,
    )
    # cdg: Resend API URL。用于配置Resend API URL，即Resend API URL。
    RESEND_API_URL: Optional[str] = Field(
        description="API URL for Resend email service",
        default=None,
    )
    # cdg: SMTP服务器主机名。用于配置SMTP服务器主机名，即SMTP服务器主机名。
    SMTP_SERVER: Optional[str] = Field(
        description="SMTP server hostname",
        default=None,
    )
    # cdg: SMTP服务器端口。用于配置SMTP服务器端口，即SMTP服务器端口。
    SMTP_PORT: Optional[int] = Field(
        description="SMTP server port number",
        default=465,
    )
    # cdg: SMTP用户名。用于配置SMTP用户名，即SMTP用户名。
    SMTP_USERNAME: Optional[str] = Field(
        description="Username for SMTP authentication",
        default=None,
    )
    # cdg: SMTP密码。用于配置SMTP密码，即SMTP密码。
    SMTP_PASSWORD: Optional[str] = Field(
        description="Password for SMTP authentication",
        default=None,
    )
    # cdg: 是否启用TLS加密。用于配置是否启用TLS加密，即是否启用TLS加密。
    SMTP_USE_TLS: bool = Field(
        description="Enable TLS encryption for SMTP connections",
        default=False,
    )
    # cdg: 是否启用 opportunistic TLS。用于配置是否启用 opportunistic TLS，即是否启用 opportunistic 是否启用TLS加密。用于配置是否启用TLS加密，即是否启用TLS加密。
    SMTP_OPPORTUNISTIC_TLS: bool = Field(
        description="Enable opportunistic TLS for SMTP connections",
        default=False,
    )
    # cdg: 邮件发送IP限制每分钟。用于配置邮件发送IP限制每分钟，即邮件发送IP限制每分钟。
    EMAIL_SEND_IP_LIMIT_PER_MINUTE: PositiveInt = Field(
        description="Maximum number of emails allowed to be sent from the same IP address in a minute",
        default=50,
    )

# cdg: RAG ETL配置信息
class RagEtlConfig(BaseSettings):
    """
    Configuration for RAG ETL processes
    """
    # cdg: ETL类型。用于配置ETL类型，即ETL类型。
    # TODO: This config is not only for rag etl, it is also for file upload, we should move it to file upload config
    ETL_TYPE: str = Field(
        description="RAG ETL type ('dify' or 'Unstructured'), default to 'dify'",
        default="dify",
    )
    # cdg: 关键词数据源类型。用于配置关键词数据源类型，即关键词数据源类型。
    KEYWORD_DATA_SOURCE_TYPE: str = Field(
        description="Data source type for keyword extraction"
        " ('database' or other supported types), default to 'database'",
        default="database",
    )
    # cdg: Unstructured API URL。用于配置Unstructured API URL，即Unstructured API URL。
    UNSTRUCTURED_API_URL: Optional[str] = Field(
        description="API URL for Unstructured.io service",
        default=None,
    )
    # cdg: Unstructured API密钥。用于配置Unstructured API密钥，即Unstructured API密钥。
    UNSTRUCTURED_API_KEY: Optional[str] = Field(
        description="API key for Unstructured.io service",
        default="",
    )
    # cdg: 是否禁用Scar分析。用于配置是否禁用Scar分析，即是否禁用Scar分析。
    SCARF_NO_ANALYTICS: Optional[str] = Field(
        description="This is about whether to disable Scarf analytics in Unstructured library.",
        default="false",
    )

# cdg: 数据集配置信息
class DataSetConfig(BaseSettings):
    """
    Configuration for dataset management
    """
    # cdg: Interval in days for dataset cleanup operations - plan: sandbox 中文意思是：数据集清理间隔天数 - 计划：沙盒
    PLAN_SANDBOX_CLEAN_DAY_SETTING: PositiveInt = Field(
        description="Interval in days for dataset cleanup operations - plan: sandbox",
        default=30,
    )
    # cdg: 数据集清理间隔天数 - 计划：专业版和团队版
    PLAN_PRO_CLEAN_DAY_SETTING: PositiveInt = Field(
        description="Interval in days for dataset cleanup operations - plan: pro and team",
        default=7,
    )
    # cdg: 是否启用数据集操作功能   
    DATASET_OPERATOR_ENABLED: bool = Field(
        description="Enable or disable dataset operator functionality",
        default=False,
    )
    # cdg: TiDB Serverless集群数量。用于配置TiDB Serverless集群数量，即TiDB Serverless集群数量。
    TIDB_SERVERLESS_NUMBER: PositiveInt = Field(
        description="number of tidb serverless cluster",
        default=500,
    )
    # cdg: 是否启用创建TiDB服务作业功能。用于配置是否启用创建TiDB服务作业功能，即是否启用创建TiDB服务作业功能。
    CREATE_TIDB_SERVICE_JOB_ENABLED: bool = Field(
        description="Enable or disable create tidb service job",
        default=False,
    )
    # cdg: 消息清理间隔天数 - 计划：沙盒
    PLAN_SANDBOX_CLEAN_MESSAGE_DAY_SETTING: PositiveInt = Field(
        description="Interval in days for message cleanup operations - plan: sandbox",
        default=30,
    )

# cdg: 工作空间配置信息
class WorkspaceConfig(BaseSettings):
    """
    Configuration for workspace management
    """
    # cdg: 工作空间邀请链接过期时间 
    INVITE_EXPIRY_HOURS: PositiveInt = Field(
        description="Expiration time in hours for workspace invitation links",
        default=72,
    )

# cdg: 知识库检索配置信息
class IndexingConfig(BaseSettings):
    """
    Configuration for indexing operations
    """
    # cdg: 知识库检索最大分段长度   
    INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH: PositiveInt = Field(
        description="Maximum token length for text segmentation during indexing",
        default=4000,
    )
    # cdg: 子块预览数量
    CHILD_CHUNKS_PREVIEW_NUMBER: PositiveInt = Field(
        description="Maximum number of child chunks to preview",
        default=50,
    )

# cdg: 多模态传输配置信息
class MultiModalTransferConfig(BaseSettings):
    MULTIMODAL_SEND_FORMAT: Literal["base64", "url"] = Field(
        description="Format for sending files in multimodal contexts ('base64' or 'url'), default is base64",
        default="base64",
    )

# cdg: Celery心跳配置信息
class CeleryBeatConfig(BaseSettings):
    CELERY_BEAT_SCHEDULER_TIME: int = Field(
        description="Interval in days for Celery Beat scheduler execution, default to 1 day",
        default=1,
    )

# cdg: 位置配置信息
class PositionConfig(BaseSettings):
    POSITION_PROVIDER_PINS: str = Field(
        description="Comma-separated list of pinned model providers",
        default="",
    )

    POSITION_PROVIDER_INCLUDES: str = Field(
        description="Comma-separated list of included model providers",
        default="",
    )

    POSITION_PROVIDER_EXCLUDES: str = Field(
        description="Comma-separated list of excluded model providers",
        default="",
    )

    POSITION_TOOL_PINS: str = Field(
        description="Comma-separated list of pinned tools",
        default="",
    )

    POSITION_TOOL_INCLUDES: str = Field(
        description="Comma-separated list of included tools",
        default="",
    )

    POSITION_TOOL_EXCLUDES: str = Field(
        description="Comma-separated list of excluded tools",
        default="",
    )

    @property
    def POSITION_PROVIDER_PINS_LIST(self) -> list[str]:
        return [item.strip() for item in self.POSITION_PROVIDER_PINS.split(",") if item.strip() != ""]

    @property
    def POSITION_PROVIDER_INCLUDES_SET(self) -> set[str]:
        return {item.strip() for item in self.POSITION_PROVIDER_INCLUDES.split(",") if item.strip() != ""}

    @property
    def POSITION_PROVIDER_EXCLUDES_SET(self) -> set[str]:
        return {item.strip() for item in self.POSITION_PROVIDER_EXCLUDES.split(",") if item.strip() != ""}

    @property
    def POSITION_TOOL_PINS_LIST(self) -> list[str]:
        return [item.strip() for item in self.POSITION_TOOL_PINS.split(",") if item.strip() != ""]

    @property
    def POSITION_TOOL_INCLUDES_SET(self) -> set[str]:
        return {item.strip() for item in self.POSITION_TOOL_INCLUDES.split(",") if item.strip() != ""}

    @property
    def POSITION_TOOL_EXCLUDES_SET(self) -> set[str]:
        return {item.strip() for item in self.POSITION_TOOL_EXCLUDES.split(",") if item.strip() != ""}

# cdg: 登录配置信息
class LoginConfig(BaseSettings):
    ENABLE_EMAIL_CODE_LOGIN: bool = Field(
        description="whether to enable email code login",
        default=False,
    )
    ENABLE_EMAIL_PASSWORD_LOGIN: bool = Field(
        description="whether to enable email password login",
        default=True,
    )
    ENABLE_SOCIAL_OAUTH_LOGIN: bool = Field(
        description="whether to enable github/google oauth login",
        default=False,
    )
    EMAIL_CODE_LOGIN_TOKEN_EXPIRY_MINUTES: PositiveInt = Field(
        description="expiry time in minutes for email code login token",
        default=5,
    )
    ALLOW_REGISTER: bool = Field(
        description="whether to enable register",
        default=False,
    )
    ALLOW_CREATE_WORKSPACE: bool = Field(
        description="whether to enable create workspace",
        default=False,
    )

# cdg: 账户配置信息
class AccountConfig(BaseSettings):
    ACCOUNT_DELETION_TOKEN_EXPIRY_MINUTES: PositiveInt = Field(
        description="Duration in minutes for which a account deletion token remains valid",
        default=5,
    )

# cdg: 功能特性配置信息，通过类继承的方式实现对上述配置信息的组合
class FeatureConfig(
    # place the configs in alphabet order
    AppExecutionConfig,
    AuthConfig,  # Changed from OAuthConfig to AuthConfig
    BillingConfig,
    CodeExecutionSandboxConfig,
    DataSetConfig,
    EndpointConfig,
    FileAccessConfig,
    FileUploadConfig,
    HttpConfig,
    InnerAPIConfig,
    IndexingConfig,
    LoggingConfig,
    MailConfig,
    ModelLoadBalanceConfig,
    ModerationConfig,
    MultiModalTransferConfig,
    PositionConfig,
    RagEtlConfig,
    SecurityConfig,
    ToolConfig,
    UpdateConfig,
    WorkflowConfig,
    WorkflowNodeExecutionConfig,
    WorkspaceConfig,
    LoginConfig,
    AccountConfig,
    # hosted services config
    HostedServiceConfig,
    CeleryBeatConfig,
):
    pass
