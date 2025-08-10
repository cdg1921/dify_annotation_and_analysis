import logging
import time

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document

# cdg: 异步删除分段索引。通过@shared_task装饰器，将delete_segment_from_index_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def delete_segment_from_index_task(index_node_ids: list, dataset_id: str, document_id: str):
    """
    Async Remove segment from index
    :param index_node_ids:
    :param dataset_id:
    :param document_id:

    Usage: delete_segment_from_index_task.delay(segment_ids)
    """
    logging.info(click.style("Start delete segment from index", fg="green"))
    start_at = time.perf_counter()
    try:
        # cdg: 根据知识库ID获取知识库。
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        # cdg: 如果知识库不存在，则返回。
        if not dataset:
            return

        # cdg: 根据文档ID获取文档。
        dataset_document = db.session.query(Document).filter(Document.id == document_id).first()
        # cdg: 如果文档不存在，则返回。
        if not dataset_document:
            return

        # cdg: 如果文档不可用、已归档或索引状态不为已完成，则返回。
        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
            return

        # cdg: 获取文档的索引类型。
        index_type = dataset_document.doc_form
        # cdg: 初始化索引处理器。
        index_processor = IndexProcessorFactory(index_type).init_index_processor()

        # cdg: 删除分段索引。
        index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

        end_at = time.perf_counter()
        logging.info(click.style("Segment deleted from index latency: {}".format(end_at - start_at), fg="green"))
    except Exception:
        logging.exception("delete segment from index failed")
