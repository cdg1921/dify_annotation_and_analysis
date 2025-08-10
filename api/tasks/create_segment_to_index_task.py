import datetime
import logging
import time
from typing import Optional

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DocumentSegment

# cdg: 异步添加文本块到向量库中。通过@shared_task装饰器，将create_segment_to_index_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def create_segment_to_index_task(segment_id: str, keywords: Optional[list[str]] = None):
    """
    Async create segment to index
    :param segment_id:
    :param keywords:
    Usage: create_segment_to_index_task.delay(segment_id)
    """
    logging.info(click.style("Start create segment to index: {}".format(segment_id), fg="green"))
    start_at = time.perf_counter()

    # cdg: 根据文本块ID获取文本块。
    segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_id).first()
    # cdg: 如果文本块不存在，则抛出异常。
    if not segment:
        raise NotFound("Segment not found")

    # cdg: 如果文本块的状态不是等待中，则返回。
    if segment.status != "waiting":
        return

    # cdg: 构建文本块的索引缓存键。
    indexing_cache_key = "segment_{}_indexing".format(segment.id)

    # cdg: 更新文本块的状态为索引中。
    try:
        # cdg: 更新文本块的状态为索引中。
        # update segment status to indexing
        update_params = {
            DocumentSegment.status: "indexing",
            DocumentSegment.indexing_at: datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
        }
        DocumentSegment.query.filter_by(id=segment.id).update(update_params)
        db.session.commit()
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            },
        )

        dataset = segment.dataset

        if not dataset:
            logging.info(click.style("Segment {} has no dataset, pass.".format(segment.id), fg="cyan"))
            return

        dataset_document = segment.document

        if not dataset_document:
            logging.info(click.style("Segment {} has no document, pass.".format(segment.id), fg="cyan"))
            return

        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
            logging.info(click.style("Segment {} document status is invalid, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 获取知识库的索引类型。
        index_type = dataset.doc_form
        # cdg: 初始化索引处理器。
        index_processor = IndexProcessorFactory(index_type).init_index_processor()

        # cdg: 将文本块添加到向量库中。
        index_processor.load(dataset, [document])

        # cdg: 更新文本块的状态为已完成。   
        # update segment to completed
        update_params = {
            DocumentSegment.status: "completed",
            DocumentSegment.completed_at: datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
        }
        DocumentSegment.query.filter_by(id=segment.id).update(update_params)
        db.session.commit()

        end_at = time.perf_counter()
        logging.info(
            click.style("Segment created to index: {} latency: {}".format(segment.id, end_at - start_at), fg="green")
        )
    except Exception as e:
        logging.exception("create segment to index failed")
        segment.enabled = False
        segment.disabled_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        segment.status = "error"
        segment.error = str(e)
        db.session.commit()
    finally:
        redis_client.delete(indexing_cache_key)
