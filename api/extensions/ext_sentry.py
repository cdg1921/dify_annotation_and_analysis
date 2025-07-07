from configs import dify_config
from dify_app import DifyApp

# cdg: 初始化Sentry
def init_app(app: DifyApp):
    if dify_config.SENTRY_DSN: # cdg: 如果SENTRY_DSN配置了，则初始化Sentry
        import openai
        import sentry_sdk
        from langfuse import parse_error  # type: ignore
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.flask import FlaskIntegration
        from werkzeug.exceptions import HTTPException

        from core.model_runtime.errors.invoke import InvokeRateLimitError

        def before_send(event, hint):
            # cdg:消息发送前处理，如果错误中包含parse_error.defaultErrorResponse，则不发送
            if "exc_info" in hint:
                exc_type, exc_value, tb = hint["exc_info"]
                if parse_error.defaultErrorResponse in str(exc_value):
                    return None

            return event
        # cdg: Sentry客户端初始化
        sentry_sdk.init(
            dsn=dify_config.SENTRY_DSN, # cdg: Sentry的DSN
            integrations=[FlaskIntegration(), CeleryIntegration()], # cdg: 集成Flask和Celery，用于捕获Flask和Celery的错误
            # cdg: 忽略的错误，如果错误在忽略的错误列表中，则不发送
            ignore_errors=[
                HTTPException,
                ValueError,
                FileNotFoundError,
                openai.APIStatusError,
                InvokeRateLimitError,
                parse_error.defaultErrorResponse,
            ],
            traces_sample_rate=dify_config.SENTRY_TRACES_SAMPLE_RATE, # cdg: 跟踪任务的采样率，用于控制跟踪任务的采样率，从而控制跟踪任务的性能
            profiles_sample_rate=dify_config.SENTRY_PROFILES_SAMPLE_RATE, # cdg: 性能分析的采样率，用于控制性能分析的采样率，从而控制性能分析的性能
            environment=dify_config.DEPLOY_ENV, # cdg: 环境，用于控制Sentry的日志级别
            release=f"dify-{dify_config.CURRENT_VERSION}-{dify_config.COMMIT_SHA}", # cdg: 版本，用于控制Sentry的日志级别
            before_send=before_send, # cdg: 消息发送前处理，如果错误中包含parse_error.defaultErrorResponse，则不发送
        )
