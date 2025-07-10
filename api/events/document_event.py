from blinker import signal

# cdg: 信号的名称是document-was-deleted，信号的接收器是document_was_deleted。
# cdg: 当document被删除时，document_was_deleted信号被触发，所有连接的接收器都会收到这个信号。
# sender: document
document_was_deleted = signal("document-was-deleted")
