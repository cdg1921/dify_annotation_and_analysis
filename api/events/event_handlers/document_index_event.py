from blinker import signal

# cdg: 使用@document_index_created.connect装饰器，将handle函数注册为文档索引创建事件的处理器。
# sender: document
document_index_created = signal("document-index-created")
