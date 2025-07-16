import json

import requests

from services.auth.api_key_auth_base import ApiKeyAuthBase

# cdg: 定义FirecrawlAuth类，继承自ApiKeyAuthBase类，用于实现API密钥认证，Firecrawl是一个可以提供API服务的开源爬虫工具，我们只需要给它一个 URL，无需提供网站地图（sitemap），它就能抓取该 URL 的当前网页或更深层的网页，并可以把抓到的数据转变成markdown格式，这种格式更适合LLM阅读
# cdg: Firecrawl的功能包括：
# cdg: Scrape：抓取 URL 当前页面的内容，可以以 markdown 格式返回
# cdg: Crawl：递归抓取 URL 的子域，并可以以 markdown 格式返回内容
# cdg: Map：可以非常快速的获取输入网站的所有 URL
# cdg: Extract：使用 LLMs 从页面中提取结构化数据
class FirecrawlAuth(ApiKeyAuthBase):
    def __init__(self, credentials: dict):
        # cdg: 调用父类ApiKeyAuthBase的__init__方法，初始化ApiKeyAuthBase类。
        super().__init__(credentials)
        auth_type = credentials.get("auth_type")
        # cdg: 如果auth_type不是bearer，则抛出异常。
        if auth_type != "bearer":
            raise ValueError("Invalid auth type, Firecrawl auth type must be Bearer")
        # cdg: 获取api_key。
        self.api_key = credentials.get("config", {}).get("api_key", None)
        # cdg: 获取base_url。
        self.base_url = credentials.get("config", {}).get("base_url", "https://api.firecrawl.dev")
        # cdg: 如果api_key为空，则抛出异常。
        if not self.api_key:
            raise ValueError("No API key provided")

    # cdg: 定义validate_credentials方法，用于验证Firecrawl的API密钥。   
    def validate_credentials(self):
        # cdg: 准备请求头。
        headers = self._prepare_headers()
        # cdg: 准备请求选项。
        options = {
            "url": "https://example.com",
            "crawlerOptions": {"excludes": [], "includes": [], "limit": 1},
            "pageOptions": {"onlyMainContent": True},
        }
        # cdg: 发送请求。
        response = self._post_request(f"{self.base_url}/v0/crawl", options, headers)
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

    # cdg: 定义_handle_error方法，用于处理错误。
    def _handle_error(self, response):
        # cdg: 如果请求状态码为402、409或500，则抛出异常。
        if response.status_code in {402, 409, 500}:
            # cdg: 获取错误信息。
            error_message = response.json().get("error", "Unknown error occurred")
            raise Exception(f"Failed to authorize. Status code: {response.status_code}. Error: {error_message}")
        else:
            # cdg: 如果请求文本不为空，则抛出异常。
            if response.text:
                # cdg: 获取错误信息。
                error_message = json.loads(response.text).get("error", "Unknown error occurred")
                raise Exception(f"Failed to authorize. Status code: {response.status_code}. Error: {error_message}")
            raise Exception(f"Unexpected error occurred while trying to authorize. Status code: {response.status_code}")
