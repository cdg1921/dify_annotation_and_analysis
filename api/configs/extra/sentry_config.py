from typing import Optional

from pydantic import Field, NonNegativeFloat
from pydantic_settings import BaseSettings

# cdg: Sentry配置信息。Sentry是一个错误监控平台，用于监控和报告应用程序的错误。
class SentryConfig(BaseSettings):
    """
    Configuration settings for Sentry error tracking and performance monitoring
    """
    # cdg: Sentry数据源名称（DSN）
    SENTRY_DSN: Optional[str] = Field(
        description="Sentry Data Source Name (DSN)."
        " This is the unique identifier of your Sentry project, used to send events to the correct project.",
        default=None,
    )
    # cdg: Sentry性能监控采样率
    SENTRY_TRACES_SAMPLE_RATE: NonNegativeFloat = Field(
        description="Sample rate for Sentry performance monitoring traces."
        " Value between 0.0 and 1.0, where 1.0 means 100% of traces are sent to Sentry.",
        default=1.0,
    )
    # cdg: Sentry分析采样率。用于配置Sentry的性能分析采样率，即Sentry收集性能分析数据的频率。
    SENTRY_PROFILES_SAMPLE_RATE: NonNegativeFloat = Field(
        description="Sample rate for Sentry profiling."
        " Value between 0.0 and 1.0, where 1.0 means 100% of profiles are sent to Sentry.",
        default=1.0,
    )
