from blinker import signal

# cdg: 信号的名称是tenant-was-created，信号的接收器是tenant_was_created。
# cdg: 当tenant被创建时，tenant_was_created信号被触发，所有连接的接收器都会收到这个信号。
# sender: tenant
tenant_was_created = signal("tenant-was-created")

# cdg: 信号的名称是tenant-was-updated，信号的接收器是tenant_was_updated。
# cdg: 当tenant被更新时，tenant_was_updated信号被触发，所有连接的接收器都会收到这个信号。
# sender: tenant
tenant_was_updated = signal("tenant-was-updated")
