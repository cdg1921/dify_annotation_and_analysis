from services.auth.api_key_auth_base import ApiKeyAuthBase
from services.auth.auth_type import AuthType

# cdg: 定义ApiKeyAuthFactory类，用于根据提供商类型创建相应的API密钥认证对象。
class ApiKeyAuthFactory:
    def __init__(self, provider: str, credentials: dict):
        auth_factory = self.get_apikey_auth_factory(provider)
        self.auth = auth_factory(credentials)

    # cdg: 定义validate_credentials方法，用于验证API密钥认证对象的API密钥。
    def validate_credentials(self):
        return self.auth.validate_credentials()

    # cdg: 定义get_apikey_auth_factory方法，用于根据提供商类型创建相应的API密钥认证对象。
    @staticmethod
    def get_apikey_auth_factory(provider: str) -> type[ApiKeyAuthBase]:
        match provider:
            case AuthType.FIRECRAWL:
                from services.auth.firecrawl.firecrawl import FirecrawlAuth

                return FirecrawlAuth
            case AuthType.JINA:
                from services.auth.jina.jina import JinaAuth

                return JinaAuth
            case _:
                raise ValueError("Invalid provider")
