import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.constant.index_type import IndexType
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import ChildDocument, Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument

# cdg: 异步启用文档片段索引。使用@shared_task装饰器将函数标记为Celery任务，并指定任务队列为"dataset"。
@shared_task(queue="dataset")
def enable_segments_to_index_task(segment_ids: list, dataset_id: str, document_id: str):
    """
    Async enable segments to index
    :param segment_ids:

    Usage: enable_segments_to_index_task.delay(segment_ids)
    """
    start_at = time.perf_counter()
    # cdg: 根据知识库ID获取知识库。如果知识库不存在，则报错。
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logging.info(click.style("Dataset {} not found, pass.".format(dataset_id), fg="cyan"))
        return

    # cdg: 根据文档ID获取文档。如果文档不存在，则报错。
    dataset_document = db.session.query(DatasetDocument).filter(DatasetDocument.id == document_id).first()
    if not dataset_document:
        logging.info(click.style("Document {} not found, pass.".format(document_id), fg="cyan"))
        return
    # cdg: 如果文档状态不合法，则跳过直接返回。
    if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
        logging.info(click.style("Document {} status is invalid, pass.".format(document_id), fg="cyan"))
        return
    # cdg: 根据文档格式获取索引处理器。
    # sync index processor
    index_processor = IndexProcessorFactory(dataset_document.doc_form).init_index_processor()
    # cdg: 根据文档片段ID获取文档片段。如果文档片段不存在，则跳过直接返回。
    segments = (
        db.session.query(DocumentSegment)
        .filter(
            DocumentSegment.id.in_(segment_ids),
            DocumentSegment.dataset_id == dataset_id,
            DocumentSegment.document_id == document_id,
        )
        .all()
    )
    if not segments:
        return
    # cdg: 遍历文档片段，创建文档对象。
    try:
        # cdg: 创建文档对象列表。
        documents = []
        for segment in segments:
            document = Document(
                page_content=segment.content,
                metadata={
                    "doc_id": segment.index_node_id,
                    "doc_hash": segment.index_node_hash,
                    "document_id": document_id,
                    "dataset_id": dataset_id,
                },
            )
            # cdg: 如果文档格式为父子索引，则处理子文档。
            if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
                child_chunks = segment.child_chunks
                if child_chunks:
                    # cdg: 创建子文档对象列表。
                    child_documents = []
                    # cdg: 遍历子文档片段。
                    for child_chunk in child_chunks:
                        # cdg: 创建子文档对象。
                        child_document = ChildDocument(
                            page_content=child_chunk.content,
                            metadata={
                                "doc_id": child_chunk.index_node_id,
                                "doc_hash": child_chunk.index_node_hash,
                                "document_id": document_id,
                                "dataset_id": dataset_id,
                            },
                        )
                        # cdg: 添加子文档对象到列表。
                        child_documents.append(child_document)
                    # cdg: 设置文档的子文档。
                    document.children = child_documents
            # cdg: 添加文档对象到列表。
            documents.append(document)
        # cdg: 保存向量索引。
        index_processor.load(dataset, documents)
        # cdg: 记录处理文档的时间。
        end_at = time.perf_counter()
        logging.info(click.style("Segments enabled to index latency: {}".format(end_at - start_at), fg="green"))
    except Exception as e:
        logging.exception("enable segments to index failed")
        # update segment error msg
        db.session.query(DocumentSegment).filter(
            DocumentSegment.id.in_(segment_ids),
            DocumentSegment.dataset_id == dataset_id,
            DocumentSegment.document_id == document_id,
        ).update(
            {
                "error": str(e),
                "status": "error",
                "disabled_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                "enabled": False,
            }
        )
        db.session.commit()
    finally:
        # cdg: 遍历文档片段，删除缓存。
        for segment in segments:
            # cdg: 设置缓存键。
            indexing_cache_key = "segment_{}_indexing".format(segment.id)
            redis_client.delete(indexing_cache_key)
