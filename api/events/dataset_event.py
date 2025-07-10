from blinker import signal

# cdg: 信号的名称是dataset-was-deleted，信号的接收器是dataset_was_deleted。
# cdg: 当dataset被删除时，dataset_was_deleted信号被触发，所有连接的接收器都会收到这个信号。
# sender: dataset
dataset_was_deleted = signal("dataset-was-deleted")
