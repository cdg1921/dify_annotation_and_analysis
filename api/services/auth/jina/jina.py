import json

import requests

from services.auth.api_key_auth_base import ApiKeyAuthBase

# cdg: 定义JinaAuth类，继承自ApiKeyAuthBase类，用于实现基于JinaAuth的API密钥认证
class JinaAuth(ApiKeyAuthBase):
    def __init__(self, credentials: dict):
        # cdg: 调用父类ApiKeyAuthBase的__init__方法，初始化ApiKeyAuthBase类。
        super().__init__(credentials)
        auth_type = credentials.get("auth_type")
        # cdg: 如果auth_type不是bearer，则抛出异常。
        if auth_type != "bearer":
            raise ValueError("Invalid auth type, Jina Reader auth type must be Bearer")
        # cdg: 获取api_key。
        self.api_key = credentials.get("config", {}).get("api_key", None)
        # cdg: 如果api_key为空，则抛出异常。
        # cdg: JinaAuth验证方式只需要api_key，不需要base_url。
        if not self.api_key:
            raise ValueError("No API key provided")
    # cdg: 定义validate_credentials方法，用于验证JinaAuth的API密钥。
    def validate_credentials(self):
        # cdg: 准备请求头。
        headers = self._prepare_headers()
        # cdg: 准备请求选项。
        options = {
            "url": "https://example.com",
        }
        # cdg: 发送请求。JinaAuth的请求体为空，所以不需要传入data参数。Firecrawl需要传入crawlerOptions等配置参数。
        response = self._post_request("https://r.jina.ai", options, headers)
        # cdg: 如果请求成功，则返回True。
        if response.status_code == 200:
            return True
        # cdg: 如果请求失败，则抛出异常。
        else:
            # cdg: 处理错误。
            self._handle_error(response)

    # cdg: 定义_prepare_headers方法，用于准备请求头。
    def _prepare_headers(self):
        # cdg: 返回请求头。
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    # cdg: 定义_post_request方法，用于发送POST请求。
    def _post_request(self, url, data, headers):
        return requests.post(url, headers=headers, json=data)
    
    # cdg: 定义_handle_error方法，用于处理错误。JinaAuth与Firecrawl的错误处理方式几乎一致。
    def _handle_error(self, response):
        if response.status_code in {402, 409, 500}:
            error_message = response.json().get("error", "Unknown error occurred")
            raise Exception(f"Failed to authorize. Status code: {response.status_code}. Error: {error_message}")
        else:
            if response.text:
                error_message = json.loads(response.text).get("error", "Unknown error occurred")
                raise Exception(f"Failed to authorize. Status code: {response.status_code}. Error: {error_message}")
            raise Exception(f"Unexpected error occurred while trying to authorize. Status code: {response.status_code}")
