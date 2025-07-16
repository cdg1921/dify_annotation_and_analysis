from services.enterprise.base import EnterpriseRequest

# cdg: 定义EnterpriseService类，根据不同的请求类型，调用不同的方法。
class EnterpriseService:
    # cdg: 定义get_info方法，获取企业信息。
    @classmethod
    def get_info(cls):
        return EnterpriseRequest.send_request("GET", "/info")

    # cdg: 定义get_app_web_sso_enabled方法，获取应用的Web SSO是否启用。
    @classmethod
    def get_app_web_sso_enabled(cls, app_code):
        return EnterpriseRequest.send_request("GET", f"/app-sso-setting?appCode={app_code}")
