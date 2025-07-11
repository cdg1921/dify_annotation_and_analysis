from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# cdg: Notion配置信息
class NotionConfig(BaseSettings):
    """
    Configuration settings for Notion integration
    """
    # cdg: Notion客户端ID
    NOTION_CLIENT_ID: Optional[str] = Field(
        description="Client ID for Notion API authentication. Required for OAuth 2.0 flow.",
        default=None,
    )
    # cdg: Notion客户端密钥
    NOTION_CLIENT_SECRET: Optional[str] = Field(
        description="Client secret for Notion API authentication. Required for OAuth 2.0 flow.",
        default=None,
    )
    # cdg: Notion集成类型
    NOTION_INTEGRATION_TYPE: Optional[str] = Field(
        description="Type of Notion integration."
        " Set to 'internal' for internal integrations, or None for public integrations.",
        default=None,
    )
    # cdg: Notion内部密钥
    NOTION_INTERNAL_SECRET: Optional[str] = Field(
        description="Secret key for internal Notion integrations. Required when NOTION_INTEGRATION_TYPE is 'internal'.",
        default=None,
    )
    # cdg: Notion集成令牌
    NOTION_INTEGRATION_TOKEN: Optional[str] = Field(
        description="Integration token for Notion API access. Used for direct API calls without OAuth flow.",
        default=None,
    )
