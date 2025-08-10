import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document, DocumentSegment

# cdg: 异步更新文档索引。通过@shared_task装饰器，将document_indexing_update_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def document_indexing_update_task(dataset_id: str, document_id: str):
    """
    Async update document
    :param dataset_id:
    :param document_id:

    Usage: document_indexing_update_task.delay(dataset_id, document_id)
    """
    logging.info(click.style("Start update document: {}".format(document_id), fg="green"))
    start_at = time.perf_counter()

    # cdg: 根据文档ID和知识库ID获取文档。
    document = db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()

    # cdg: 如果文档不存在，则报错
    if not document:
        raise NotFound("Document not found")

    # cdg: 更新文档索引状态和处理开始时间。
    document.indexing_status = "parsing"
    document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    db.session.commit()

    # delete all document segment and index
    try:
        # cdg: 根据知识库ID获取知识库。如果知识库不存在，则报错。
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise Exception("Dataset not found")

        # cdg: 根据文档格式获取索引处理器。
        index_type = document.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        # cdg: 获取文档片段。
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
        if segments:
            index_node_ids = [segment.index_node_id for segment in segments]

            # cdg: 从向量索引中删除文档片段。
            # delete from vector index
            index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

            # cdg: 删除文档片段。
            for segment in segments:
                db.session.delete(segment)

            # cdg: 提交事务。
            db.session.commit()

        # cdg: 记录处理文档的时间。
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Cleaned document when document update data source or process rule: {} latency: {}".format(
                    document_id, end_at - start_at
                ),
                fg="green",
            )
        )
    except Exception:
        logging.exception("Cleaned document when document update data source or process rule failed")

    try:
        indexing_runner = IndexingRunner()
        indexing_runner.run([document])
        end_at = time.perf_counter()
        logging.info(click.style("update document: {} latency: {}".format(document.id, end_at - start_at), fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        pass
