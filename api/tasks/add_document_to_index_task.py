import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.index_processor.constant.index_type import IndexType
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import ChildDocument, Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DatasetAutoDisableLog, DocumentSegment
from models.dataset import Document as DatasetDocument

# cdg: 异步添加文档到向量库中。通过@shared_task装饰器，将add_document_to_index_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def add_document_to_index_task(dataset_document_id: str):
    """
    Async Add document to index
    :param dataset_document_id:

    Usage: add_document_to_index.delay(dataset_document_id)
    """
    logging.info(click.style("Start add document to index: {}".format(dataset_document_id), fg="green"))
    start_at = time.perf_counter()

    # cdg: 获取文档。
    dataset_document = db.session.query(DatasetDocument).filter(DatasetDocument.id == dataset_document_id).first()
    if not dataset_document:
        raise NotFound("Document not found")

    # cdg: 如果文档的索引状态不是已完成，则返回。
    if dataset_document.indexing_status != "completed":
        return

    # cdg: 构建文档的索引缓存键。
    indexing_cache_key = "document_{}_indexing".format(dataset_document.id)

    try:
        # cdg: 根据文档ID获取文档的分段。
        segments = (
            db.session.query(DocumentSegment)
            .filter(
                DocumentSegment.document_id == dataset_document.id,
                DocumentSegment.enabled == False,
                DocumentSegment.status == "completed",
            )
            .order_by(DocumentSegment.position.asc())
            .all()
        )

        documents = []
        for segment in segments:
            # cdg: 构建文本分段对象。包含文本分段的内容、元数据等信息。
            # cdg: 注意，在这个for循环中，documents列表中存储的是文本分段对象。
            document = Document(
                page_content=segment.content,
                metadata={
                    "doc_id": segment.index_node_id,
                    "doc_hash": segment.index_node_hash,
                    "document_id": segment.document_id,
                    "dataset_id": segment.dataset_id,
                },
            )
            # cdg: 如果文档的索引方式为父子分段模式，则构建子块的文档对象。
            if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
                # cdg: 获取文档的子块。
                child_chunks = segment.child_chunks
                if child_chunks:
                    # cdg: 构建子块的文档对象。
                    child_documents = []
                    for child_chunk in child_chunks:
                        # cdg: 构建子块的文档对象。
                        child_document = ChildDocument(
                            page_content=child_chunk.content,
                            metadata={
                                "doc_id": child_chunk.index_node_id,
                                "doc_hash": child_chunk.index_node_hash,
                                "document_id": segment.document_id,
                                "dataset_id": segment.dataset_id,
                            },
                        )
                        # cdg: 将子块的文档对象添加到子文档列表中。
                        child_documents.append(child_document)
                        
                    document.children = child_documents
            documents.append(document)

        # cdg: 获取文档所属的知识库。
        dataset = dataset_document.dataset

        # cdg: 如果文档没有所属的知识库，则抛出异常。
        if not dataset:
            raise Exception("Document has no dataset")

        # cdg: 获取文档的索引方式。
        index_type = dataset.doc_form
        # cdg: 初始化索引处理器。
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        # cdg: 将文档添加到向量库中。
        index_processor.load(dataset, documents)

        # cdg: 删除文档的自动禁用日志。
        # delete auto disable log
        db.session.query(DatasetAutoDisableLog).filter(
            DatasetAutoDisableLog.document_id == dataset_document.id
        ).delete()

        # cdg: 更新文档的索引状态为已完成。
        # update segment to enable
        db.session.query(DocumentSegment).filter(DocumentSegment.document_id == dataset_document.id).update(
            {
                DocumentSegment.enabled: True,
                DocumentSegment.disabled_at: None,
                DocumentSegment.disabled_by: None,
                DocumentSegment.updated_at: datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
            }
        )
        db.session.commit()

        # cdg: 计算添加文档到向量库中的时间，并记录日志。
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Document added to index: {} latency: {}".format(dataset_document.id, end_at - start_at), fg="green"
            )
        )
    except Exception as e:
        logging.exception("add document to index failed")
        dataset_document.enabled = False
        dataset_document.disabled_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        dataset_document.status = "error"
        dataset_document.error = str(e)
        db.session.commit()
    finally:
        # cdg: 删除Redis中文档的索引缓存。
        redis_client.delete(indexing_cache_key)
