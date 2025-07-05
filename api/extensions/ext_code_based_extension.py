from core.extension.extension import Extension
from dify_app import DifyApp

# cdg: 初始化代码扩展
def init_app(app: DifyApp):
    code_based_extension.init()

# cdg: 代码扩展 Extension是Dify的扩展机制，用于在Dify中添加自定义功能。
code_based_extension = Extension()
