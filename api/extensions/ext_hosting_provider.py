from core.hosting_configuration import HostingConfiguration

hosting_configuration = HostingConfiguration()


from dify_app import DifyApp

# cdg: 初始化远程服务提供者，这个函数的作用是初始化远程服务提供者，用于在Dify中使用远程服务提供者。
def init_app(app: DifyApp):
    hosting_configuration.init_app(app)
