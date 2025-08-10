import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from configs import dify_config
from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from extensions.ext_database import db
from models.dataset import Dataset, Document
from services.feature_service import FeatureService

# cdg: 异步处理文档索引。通过@shared_task装饰器，将document_indexing_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: document_indexing_task.delay(dataset_id, document_id)
    """
    documents = []
    start_at = time.perf_counter()

    # cdg: 根据知识库ID获取知识库。
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    # cdg: 如果知识库不存在，则记录日志，并返回空值。
    if not dataset:
        logging.info(click.style("Dataset is not found: {}".format(dataset_id), fg="yellow"))
        return
    # cdg: 获取知识库的特征。
    # check document limit
    features = FeatureService.get_features(dataset.tenant_id)
    try:
        # cdg: 如果知识库的计费功能启用，则检查文档数量和向量空间限制。
        if features.billing.enabled:
            # cdg: 获取向量空间。
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
                document.stopped_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                db.session.add(document)
        db.session.commit()
        return

    # cdg: 遍历文档ID。
    for document_id in document_ids:
        logging.info(click.style("Start process document: {}".format(document_id), fg="green"))

        document = (
            db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
        )

        if document:
            document.indexing_status = "parsing"
            document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
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
