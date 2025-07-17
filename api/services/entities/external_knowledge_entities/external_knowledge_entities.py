from typing import Literal, Optional, Union

from pydantic import BaseModel

# cdg: 定义AuthorizationConfig类，用于表示外部知识库授权配置。
class AuthorizationConfig(BaseModel):
    type: Literal[None, "basic", "bearer", "custom"]
    api_key: Union[None, str] = None
    header: Union[None, str] = None

# cdg: 定义Authorization类，用于表示外部知识库授权信息。
class Authorization(BaseModel):
    type: Literal["no-auth", "api-key"]
    config: Optional[AuthorizationConfig] = None

# cdg: 定义ProcessStatusSetting类，用于表示外部知识库处理状态设置。
class ProcessStatusSetting(BaseModel):
    request_method: str
    url: str

# cdg: 定义ExternalKnowledgeApiSetting类，用于表示外部知识库API设置。
class ExternalKnowledgeApiSetting(BaseModel):
    url: str
    request_method: str
    headers: Optional[dict] = None
    params: Optional[dict] = None
