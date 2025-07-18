from pydantic import Field
from pydantic_settings import BaseSettings

# cdg:OpenTelemetry，简称 OTel，是一个供应商中立的开源可观测性框架，用于仪表化、生成、收集和导出诸如跟踪、度量、日志等遥测数据。作为行业标准，它得到了40多个可观测性供应商的支持，在许多库、服务和应用中得到集成，并得到了众多最终用户的采用。
class OTelConfig(BaseSettings):
    """
    OpenTelemetry configuration settings
    """

    ENABLE_OTEL: bool = Field(
        description="Whether to enable OpenTelemetry",
        default=False,
    )

    OTLP_BASE_ENDPOINT: str = Field(
        description="OTLP base endpoint",
        default="http://localhost:4318",
    )

    OTLP_API_KEY: str = Field(
        description="OTLP API key",
        default="",
    )

    OTEL_EXPORTER_TYPE: str = Field(
        description="OTEL exporter type",
        default="otlp",
    )

    OTEL_EXPORTER_OTLP_PROTOCOL: str = Field(
        description="OTLP exporter protocol ('grpc' or 'http')",
        default="http",
    )

    OTEL_SAMPLING_RATE: float = Field(default=0.1, description="Sampling rate for traces (0.0 to 1.0)")

    OTEL_BATCH_EXPORT_SCHEDULE_DELAY: int = Field(
        default=5000, description="Batch export schedule delay in milliseconds"
    )

    OTEL_MAX_QUEUE_SIZE: int = Field(default=2048, description="Maximum queue size for the batch span processor")

    OTEL_MAX_EXPORT_BATCH_SIZE: int = Field(default=512, description="Maximum export batch size")

    OTEL_METRIC_EXPORT_INTERVAL: int = Field(default=60000, description="Metric export interval in milliseconds")

    OTEL_BATCH_EXPORT_TIMEOUT: int = Field(default=10000, description="Batch export timeout in milliseconds")

    OTEL_METRIC_EXPORT_TIMEOUT: int = Field(default=30000, description="Metric export timeout in milliseconds")
