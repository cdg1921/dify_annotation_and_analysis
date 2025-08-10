import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DocumentSegment

# cdg: 异步禁用分段索引。通过@shared_task装饰器，将disable_segment_from_index_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def disable_segment_from_index_task(segment_id: str):
    """
    Async disable segment from index
    :param segment_id:

    Usage: disable_segment_from_index_task.delay(segment_id)
    """
    logging.info(click.style("Start disable segment from index: {}".format(segment_id), fg="green"))
    start_at = time.perf_counter()

    # cdg: 根据分段ID获取分段。
    segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_id).first()
    # cdg: 如果分段不存在，则抛出异常。
    if not segment:
        raise NotFound("Segment not found")

    # cdg: 如果分段状态不为已完成，则抛出异常。
    if segment.status != "completed":
        raise NotFound("Segment is not completed , disable action is not allowed.")

    # cdg: 构建分段的索引缓存键。
    indexing_cache_key = "segment_{}_indexing".format(segment.id)

    # cdg: 尝试禁用分段索引。
    try:
        # cdg: 获取分段所属的知识库。
        dataset = segment.dataset
        # cdg: 如果分段没有所属的知识库，则记录日志。
        if not dataset:
            logging.info(click.style("Segment {} has no dataset, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 获取分段所属的文档。
        dataset_document = segment.document
        # cdg: 如果分段没有所属的文档，则记录日志。 
        if not dataset_document:
            logging.info(click.style("Segment {} has no document, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 如果文档不可用、已归档或索引状态不为已完成，则记录日志。
        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
            logging.info(click.style("Segment {} document status is invalid, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 获取文档的索引类型。
        index_type = dataset_document.doc_form
        # cdg: 初始化索引处理器。
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        # cdg: 删除分段索引。
        index_processor.clean(dataset, [segment.index_node_id])

        end_at = time.perf_counter()
        # cdg: 记录禁用分段索引的时间。
        logging.info(
            click.style("Segment removed from index: {} latency: {}".format(segment.id, end_at - start_at), fg="green")
        )
    except Exception:
        logging.exception("remove segment from index failed")
        segment.enabled = True
        db.session.commit()
    finally:
        # cdg: 删除redis中分段的索引缓存。
        redis_client.delete(indexing_cache_key)
