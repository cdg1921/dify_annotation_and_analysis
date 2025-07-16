from abc import ABC, abstractmethod

# cdg: 定义ApiKeyAuthBase类，继承自ABC类，用于实现API密钥认证的基类。
class ApiKeyAuthBase(ABC):
    def __init__(self, credentials: dict):
        self.credentials = credentials

    @abstractmethod
    def validate_credentials(self):
        raise NotImplementedError
