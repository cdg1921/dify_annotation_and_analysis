from typing import Optional

from configs import dify_config
from core.mcp.types import (
    OAuthClientInformation,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthTokens,
)
from models.tools import MCPToolProvider
from services.tools.mcp_tools_mange_service import MCPToolManageService

LATEST_PROTOCOL_VERSION = "1.0"

# cdg:定义OAuth客户端提供者，包含MCP提供者、重定向URL、客户端元数据、客户端信息、令牌、代码验证器
class OAuthClientProvider:
    mcp_provider: MCPToolProvider

    def __init__(self, provider_id: str, tenant_id: str, for_list: bool = False):
        if for_list:
            self.mcp_provider = MCPToolManageService.get_mcp_provider_by_provider_id(provider_id, tenant_id)
        else:
            self.mcp_provider = MCPToolManageService.get_mcp_provider_by_server_identifier(provider_id, tenant_id)

    # cdg:重定向URL，使用Dify的控制台API URL，并添加/console/api/mcp/oauth/callback作为重定向URL
    @property
    def redirect_url(self) -> str:
        """The URL to redirect the user agent to after authorization."""
        return dify_config.CONSOLE_API_URL + "/console/api/mcp/oauth/callback"

    # cdg:加载客户端元数据，使用重定向URL，并返回OAuth客户端元数据
    @property
    def client_metadata(self) -> OAuthClientMetadata:
        """Metadata about this OAuth client."""
        return OAuthClientMetadata(
            redirect_uris=[self.redirect_url],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            client_name="Dify",
            client_uri="https://github.com/langgenius/dify",
        )
        
    # cdg:加载客户端信息，使用MCP提供者的解密凭证，获取客户端信息，如果客户端信息不存在，则返回None
    def client_information(self) -> Optional[OAuthClientInformation]:
        """Loads information about this OAuth client."""
        client_information = self.mcp_provider.decrypted_credentials.get("client_information", {})
        if not client_information:
            return None
        return OAuthClientInformation.model_validate(client_information)
    # cdg:保存客户端信息，使用MCP提供者的解密凭证，更新客户端信息
    def save_client_information(self, client_information: OAuthClientInformationFull) -> None:
        """Saves client information after dynamic registration."""
        MCPToolManageService.update_mcp_provider_credentials(
            self.mcp_provider,
            {"client_information": client_information.model_dump()},
        )
    # cdg:加载令牌，使用MCP提供者的解密凭证，获取令牌，如果令牌不存在，则返回None
    def tokens(self) -> Optional[OAuthTokens]:
        """Loads any existing OAuth tokens for the current session."""
        credentials = self.mcp_provider.decrypted_credentials
        if not credentials:
            return None
        return OAuthTokens(
            access_token=credentials.get("access_token", ""),
            token_type=credentials.get("token_type", "Bearer"),
            expires_in=int(credentials.get("expires_in", "3600") or 3600),
            refresh_token=credentials.get("refresh_token", ""),
        )
    # cdg:保存令牌，使用MCP提供者的解密凭证，更新令牌
    def save_tokens(self, tokens: OAuthTokens) -> None:
        """Stores new OAuth tokens for the current session."""
        # update mcp provider credentials
        token_dict = tokens.model_dump()
        MCPToolManageService.update_mcp_provider_credentials(self.mcp_provider, token_dict, authed=True)
    # cdg:保存代码验证器，使用MCP提供者的解密凭证，更新代码验证器
    def save_code_verifier(self, code_verifier: str) -> None:
        """Saves a PKCE code verifier for the current session."""
        MCPToolManageService.update_mcp_provider_credentials(self.mcp_provider, {"code_verifier": code_verifier})
    # cdg:加载代码验证器，使用MCP提供者的解密凭证，获取代码验证器，如果代码验证器不存在，则返回空字符串
    def code_verifier(self) -> str:
        """Loads the PKCE code verifier for the current session."""
        # get code verifier from mcp provider credentials
        return str(self.mcp_provider.decrypted_credentials.get("code_verifier", ""))
