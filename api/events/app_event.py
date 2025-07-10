from blinker import signal
# cdg: blinker是一个Python的信号库，常用于实现“事件驱动”或“发布-订阅”模式。它允许不同部分的代码通过信号（signal）进行解耦通信。
# cdg: 其核心思想和工作原理包括：
# 1. 定义信号（signal）：创建一个信号对象，用于表示一个事件。用户可以通过 blinker.signal(name) 创建一个信号对象。信号对象代表一个事件，比如 app-was-created。
# 2. 连接信号和接收器（receiver）：将信号与处理该事件的函数或方法连接起来。用户可以通过 signal.connect(receiver) 将信号与接收器连接起来。
# 3. 发送信号：在适当的时候触发信号，通知所有连接的接收器。用户可以通过 signal.send(sender, **kwargs) 触发信号。
# 4. 接收信号：在接收器中处理信号事件。用户可以通过receiver(sender, **kwargs) 接收信号。

# cdg: 信号的名称是 app-was-created，信号的接收器是 app_was_created。
# cdg: 当 app 被创建时，app_was_created 信号被触发，所有连接的接收器都会收到这个信号。
# cdg: 接收器可以是一个函数或方法，也可以是一个对象的方法。
# cdg: 接收器可以是一个函数或方法，也可以是一个对象的方法。
# cdg: 下同。
# sender: app
app_was_created = signal("app-was-created")

# cdg: 信号的名称是app-model-config-was-updated，信号的接收器是 app_model_config_was_updated。
# cdg: 当 app 的模型配置被更新时，app_model_config_was_updated 信号被触发，所有连接的接收器都会收到这个信号。
# sender: app, kwargs: app_model_config
app_model_config_was_updated = signal("app-model-config-was-updated")

# cdg: 信号的名称是app-published-workflow-was-updated，信号的接收器是 app_published_workflow_was_updated。
# sender: app, kwargs: published_workflow
app_published_workflow_was_updated = signal("app-published-workflow-was-updated")

# cdg: 信号的名称是app-draft-workflow-was-synced，信号的接收器是 app_draft_workflow_was_synced。
# sender: app, kwargs: synced_draft_workflow
app_draft_workflow_was_synced = signal("app-draft-workflow-was-synced")
