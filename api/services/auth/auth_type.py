from enum import StrEnum

# cdg: 定义AuthType枚举类，用于表示API密钥认证的类型，目前支持firecrawl和jina两种类型。
class AuthType(StrEnum):
    FIRECRAWL = "firecrawl"
    JINA = "jinareader"
