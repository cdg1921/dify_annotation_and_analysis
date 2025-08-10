import logging
import time

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument

# cdg: 异步禁用分段索引。通过@shared_task装饰器，将disable_segments_from_index_task函数注册为Celery任务，并指定任务队列为dataset。
# cdg: 与disable_segment_from_index_task函数类似，但可以禁用多个分段。
@shared_task(queue="dataset")
def disable_segments_from_index_task(segment_ids: list, dataset_id: str, document_id: str):
    """
    Async disable segments from index
    :param segment_ids:

    Usage: disable_segments_from_index_task.delay(segment_ids, dataset_id, document_id)
    """
    start_at = time.perf_counter()

    # cdg: 根据知识库ID获取知识库。
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    # cdg: 如果知识库不存在，则记录日志，并返回空值。
    if not dataset:
        logging.info(click.style("Dataset {} not found, pass.".format(dataset_id), fg="cyan"))
        return

    # cdg: 根据文档ID获取文档。
    dataset_document = db.session.query(DatasetDocument).filter(DatasetDocument.id == document_id).first()
    # cdg: 如果文档不存在，则记录日志，并返回空值。
    if not dataset_document:
        logging.info(click.style("Document {} not found, pass.".format(document_id), fg="cyan"))
        return
    # cdg: 检查文档状态，如果文档不可用、已归档或索引状态不为已完成，则记录日志，并返回空值。
    if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
        logging.info(click.style("Document {} status is invalid, pass.".format(document_id), fg="cyan"))
        return

    # cdg: 初始化索引处理器。
    # sync index processor
    index_processor = IndexProcessorFactory(dataset_document.doc_form).init_index_processor()

    # cdg: 根据分段ID获取分段。
    segments = (
        db.session.query(DocumentSegment)
        .filter(
            DocumentSegment.id.in_(segment_ids),
            DocumentSegment.dataset_id == dataset_id,
            DocumentSegment.document_id == document_id,
        )
        .all()
    )
    # cdg: 如果分段不存在，则返回空值。
    if not segments:
        return

    try:
        # cdg: 获取分段索引节点ID。
        index_node_ids = [segment.index_node_id for segment in segments]
        # cdg: 删除分段索引。
        index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=False)
        # cdg: 记录禁用分段索引的时间。
        end_at = time.perf_counter()
        logging.info(click.style("Segments removed from index latency: {}".format(end_at - start_at), fg="green"))
    except Exception:
        # update segment error msg
        db.session.query(DocumentSegment).filter(
            DocumentSegment.id.in_(segment_ids),
            DocumentSegment.dataset_id == dataset_id,
            DocumentSegment.document_id == document_id,
        ).update(
            {
                "disabled_at": None,
                "disabled_by": None,
                "enabled": True,
            }
        )
        db.session.commit()
    finally:
        # cdg: 删除redis中分段的索引缓存。
        for segment in segments:
            indexing_cache_key = "segment_{}_indexing".format(segment.id)
            redis_client.delete(indexing_cache_key)
