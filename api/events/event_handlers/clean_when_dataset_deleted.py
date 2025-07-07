from events.dataset_event import dataset_was_deleted
from tasks.clean_dataset_task import clean_dataset_task

# cdg: 使用@dataset_was_deleted.connect装饰器，将handle函数注册为数据集删除事件的处理器。
@dataset_was_deleted.connect
def handle(sender, **kwargs): # cdg: 处理数据集删除事件，sender是数据集对象，kwargs是事件参数
    dataset = sender # cdg: 获取数据集对象
    clean_dataset_task.delay( # cdg: 延迟执行清理数据集任务
        dataset.id, # cdg: 数据集ID
        dataset.tenant_id, # cdg: 租户ID
        dataset.indexing_technique, # cdg: 索引方式
        dataset.index_struct, # cdg: 索引结构
        dataset.collection_binding_id,
        dataset.doc_form,
    )
