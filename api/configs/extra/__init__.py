from configs.extra.notion_config import NotionConfig
from configs.extra.sentry_config import SentryConfig

# cdg: 额外服务配置信息，包括Notion（知识库）和Sentry（错误监控）
class ExtraServiceConfig(
    # place the configs in alphabet order
    NotionConfig,
    SentryConfig,
):
    pass
