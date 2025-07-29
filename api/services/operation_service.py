import os

import requests

# cdg:操作服务，主要用于记录UTM信息和请求账单信息。
class OperationService:
    base_url = os.environ.get("BILLING_API_URL", "BILLING_API_URL")
    secret_key = os.environ.get("BILLING_API_SECRET_KEY", "BILLING_API_SECRET_KEY")

    # cdg:请求账单API，具体实现思路：
    # 1. 构建请求头
    # 2. 构建请求URL
    # 3. 发送请求
    # 4. 返回响应结果
    @classmethod
    def _send_request(cls, method, endpoint, json=None, params=None):
        headers = {"Content-Type": "application/json", "Billing-Api-Secret-Key": cls.secret_key}

        url = f"{cls.base_url}{endpoint}"
        response = requests.request(method, url, json=json, params=params, headers=headers)

        return response.json()

    @classmethod
    # cdg:记录UTM信息，具体实现思路：
    # 1. 构建请求参数
    # 2. 发送POST请求
    # 3. 返回响应结果
    def record_utm(cls, tenant_id: str, utm_info: dict):
        params = {
            "tenant_id": tenant_id,
            "utm_source": utm_info.get("utm_source", ""),
            "utm_medium": utm_info.get("utm_medium", ""),
            "utm_campaign": utm_info.get("utm_campaign", ""),
            "utm_content": utm_info.get("utm_content", ""),
            "utm_term": utm_info.get("utm_term", ""),
        }
        return cls._send_request("POST", "/tenant_utms", params=params)
