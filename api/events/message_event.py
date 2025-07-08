from blinker import signal

# cdg: 信号的名称是message-was-created，信号的接收器是message_was_created。
# cdg: 当message被创建时，message_was_created信号被触发，所有连接的接收器都会收到这个信号。
# sender: message, kwargs: conversation
message_was_created = signal("message-was-created")
