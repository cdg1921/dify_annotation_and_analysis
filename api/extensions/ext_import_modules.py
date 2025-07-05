from dify_app import DifyApp

# cdg: 事件处理模块初始化，这个函数的作用是初始化事件处理模块，用于在Dify中使用事件处理模块。
def init_app(app: DifyApp):
    from events import event_handlers  # noqa: F401
