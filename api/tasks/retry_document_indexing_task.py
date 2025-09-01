import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from core.indexing_runner import IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset, Document, DocumentSegment
from services.feature_service import FeatureService


# cdg: 用于异步重试文档索引. 用法：celery -A celery_app.celery_app.celery worker -Q dataset -n worker1@%h
@shared_task(queue="dataset")
def retry_document_indexing_task(dataset_id: str, document_ids: list[str]):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: retry_document_indexing_task.delay(dataset_id, document_id)
    """
    documents: list[Document] = []
    start_at = time.perf_counter()
    # cdg: 获取数据集
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
    # cdg: 遍历文档ID
    for document_id in document_ids:
        retry_indexing_cache_key = "document_{}_is_retried".format(document_id)
        # check document limit
        # cdg: 检查文档账单是否超过限制
        features = FeatureService.get_features(dataset.tenant_id)
        try:
            # cdg: 检查账单是否启用
            if features.billing.enabled:
                vector_space = features.vector_space
                # cdg: 检查向量空间是否超过限制
                if 0 < vector_space.limit <= vector_space.size:
                    raise ValueError(
                        "Your total number of documents plus the number of uploads have over the limit of "
                        "your subscription."
                    )
        except Exception as e:
            document = (
                db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
            )
            if document:
                document.indexing_status = "error"
                document.error = str(e)
                document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.add(document)
                db.session.commit()
            # cdg: 删除文档索引重试缓存
            redis_client.delete(retry_indexing_cache_key)
            return

        logging.info(click.style("Start retry document: {}".format(document_id), fg="green"))
        document = (
            db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
        )
        if not document:
            logging.info(click.style("Document not found: {}".format(document_id), fg="yellow"))
            return
        try:
            # cdg: 初始化索引处理器
            # clean old data
            index_processor = IndexProcessorFactory(document.doc_form).init_index_processor()
            # cdg: 获取文档片段
            segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
            if segments:
                # cdg: 获取文档片段的索引节点ID
                index_node_ids = [segment.index_node_id for segment in segments]
                # cdg: 从向量索引中删除文档片段
                # delete from vector index
                index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

            # cdg: 删除文档片段
            for segment in segments:
                db.session.delete(segment)
            db.session.commit()
            # cdg: 更新文档状态为parsing（解析中）
            document.indexing_status = "parsing"
            document.processing_started_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.add(document)
            db.session.commit()

            # cdg: 初始化索引运行器
            indexing_runner = IndexingRunner()
            # cdg: 运行文档索引
            indexing_runner.run([document])
            # cdg: 删除文档索引重试缓存
            redis_client.delete(retry_indexing_cache_key)
        except Exception as ex:
            document.indexing_status = "error"
            document.error = str(ex)
            document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.add(document)
            db.session.commit()
            logging.info(click.style(str(ex), fg="yellow"))
            redis_client.delete(retry_indexing_cache_key)
            pass
    end_at = time.perf_counter()
    logging.info(click.style("Retry dataset: {} latency: {}".format(dataset_id, end_at - start_at), fg="green"))
