from events.document_event import document_was_deleted
from tasks.clean_document_task import clean_document_task

# cdg: 使用@document_was_deleted.connect装饰器，将handle函数注册为文档删除事件的处理器。
@document_was_deleted.connect
def handle(sender, **kwargs): # cdg: 处理文档删除事件，sender是文档对象，kwargs是事件参数   
    document_id = sender
    dataset_id = kwargs.get("dataset_id")
    doc_form = kwargs.get("doc_form")
    file_id = kwargs.get("file_id")
    clean_document_task.delay(document_id, dataset_id, doc_form, file_id)
