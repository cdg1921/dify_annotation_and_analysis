from typing import Optional

from werkzeug.exceptions import HTTPException

# cdg: 定义基础HTTP异常类
class BaseHTTPException(HTTPException):
    error_code: str = "unknown"
    data: Optional[dict] = None

    def __init__(self, description=None, response=None): # cdg: 初始化异常
        super().__init__(description, response) # cdg: 调用父类初始化

        # cdg: 设置异常数据
        self.data = {
            "code": self.error_code,
            "message": self.description,
            "status": self.code,
        }
