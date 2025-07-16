import os

import requests

# cdg: 定义EnterpriseRequest类，用于发送企业请求。
class EnterpriseRequest:
    base_url = os.environ.get("ENTERPRISE_API_URL", "ENTERPRISE_API_URL")
    secret_key = os.environ.get("ENTERPRISE_API_SECRET_KEY", "ENTERPRISE_API_SECRET_KEY")

    # cdg: 定义proxies属性，用于设置代理。
    proxies = {
        "http": "",
        "https": "",
    }

    # cdg: 定义send_request方法，用于发送企业请求。
    @classmethod
    def send_request(cls, method, endpoint, json=None, params=None):
        # cdg: 准备请求头。
        headers = {"Content-Type": "application/json", "Enterprise-Api-Secret-Key": cls.secret_key}
        # cdg: 准备请求URL。
        url = f"{cls.base_url}{endpoint}"
        # cdg: 发送请求。
        response = requests.request(method, url, json=json, params=params, headers=headers, proxies=cls.proxies)
        return response.json()
