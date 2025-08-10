import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from configs import dify_config
from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document, DocumentSegment
from services.feature_service import FeatureService

# cdg: 异步处理文档。使用@shared_task装饰器将函数标记为Celery任务，并指定任务队列为"dataset"。
@shared_task(queue="dataset")
def duplicate_document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: duplicate_document_indexing_task.delay(dataset_id, document_id)
    """
    documents = []
    start_at = time.perf_counter()

    # cdg: 根据知识库ID获取知识库。如果知识库不存在，则报错。
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise ValueError("Dataset not found")

    # cdg: 检查文档数量是否超过限制。
    features = FeatureService.get_features(dataset.tenant_id)
    try:
        # cdg: 如果账单功能启用，则检查文档数量是否超过限制。
        if features.billing.enabled:
            vector_space = features.vector_space
            count = len(document_ids)
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")
            if 0 < vector_space.limit <= vector_space.size:
                raise ValueError(
                    "Your total number of documents plus the number of uploads have over the limit of "
                    "your subscription."
                )
    except Exception as e:
        for document_id in document_ids:
            document = (
                db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
            )
            if document:
                document.indexing_status = "error"
                document.error = str(e)
                document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.add(document)
        db.session.commit()
        return

    # cdg: 遍历文档ID列表，处理每个文档。
    for document_id in document_ids:
        logging.info(click.style("Start process document: {}".format(document_id), fg="green"))

        # cdg: 根据文档ID和知识库ID获取文档。如果文档不存在，则跳过。
        document = (
            db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
        )
        # cdg: 如果文档存在，则处理文档。
        if document:
            # clean old data
            # cdg: 根据文档格式获取索引处理器。
            index_type = document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()
            # cdg: 获取文档片段。
            segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
            # cdg: 如果文档片段存在，则处理文档片段。
            if segments:
                # cdg: 获取文档片段的索引节点ID。
                index_node_ids = [segment.index_node_id for segment in segments]

                # cdg: 从向量索引中删除文档片段。
                # delete from vector index
                index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

                # cdg: 删除数据库中文档片段。
                for segment in segments:
                    db.session.delete(segment)

                # cdg: 提交事务。
                db.session.commit()

            document.indexing_status = "parsing"
            document.processing_started_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            documents.append(document)
            db.session.add(document)
    db.session.commit()

    try:
        # cdg: 初始化索引运行器。   
        indexing_runner = IndexingRunner()
        # cdg: 运行索引。
        indexing_runner.run(documents)
        # cdg: 记录处理文档的时间。
        end_at = time.perf_counter()
        logging.info(click.style("Processed dataset: {} latency: {}".format(dataset_id, end_at - start_at), fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        pass
