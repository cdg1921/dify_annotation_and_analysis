import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Document, DocumentSegment

# cdg: 用于异步删除文档索引. celery -A celery_app.celery_app.celery worker -Q dataset -n worker1@%h
@shared_task(queue="dataset")
def remove_document_from_index_task(document_id: str):
    """
    Async Remove document from index
    :param document_id: document id

    Usage: remove_document_from_index.delay(document_id)
    """
    logging.info(click.style("Start remove document segments from index: {}".format(document_id), fg="green"))
    start_at = time.perf_counter()

    document = db.session.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise NotFound("Document not found")

    if document.indexing_status != "completed":
        return
    # cdg: 拼接文档的索引缓存键
    indexing_cache_key = "document_{}_indexing".format(document.id)

    try:
        dataset = document.dataset

        if not dataset:
            raise Exception("Document has no dataset")
        # cdg: 初始化索引处理器
        index_processor = IndexProcessorFactory(document.doc_form).init_index_processor()
        # cdg: 获取文档片段
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document.id).all()
        # cdg: 获取文档片段的索引节点ID
        index_node_ids = [segment.index_node_id for segment in segments]
        if index_node_ids:
            try:
                # cdg: 清理文档片段的索引
                index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=False)
            except Exception:
                logging.exception(f"clean dataset {dataset.id} from index failed")
        # update segment to disable
        # cdg: 禁用文档片段
        db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document.id).update(
            {
                DocumentSegment.enabled: False,
                DocumentSegment.disabled_at: datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
                DocumentSegment.disabled_by: document.disabled_by,
                DocumentSegment.updated_at: datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
            }
        )
        db.session.commit()

        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Document removed from index: {} latency: {}".format(document.id, end_at - start_at), fg="green"
            )
        )
    except Exception:
        logging.exception("remove document from index failed")
        if not document.archived:
            document.enabled = True
            db.session.commit()
    finally:
        # cdg: 删除文档的索引缓存
        redis_client.delete(indexing_cache_key)
