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
from models.dataset import DocumentSegment

# cdg: 异步启用文档片段索引。使用@shared_task装饰器将函数标记为Celery任务，并指定任务队列为"dataset"。
@shared_task(queue="dataset")
def enable_segment_to_index_task(segment_id: str):
    """
    Async enable segment to index
    :param segment_id:

    Usage: enable_segment_to_index_task.delay(segment_id)
    """
    logging.info(click.style("Start enable segment to index: {}".format(segment_id), fg="green"))
    start_at = time.perf_counter()

    # cdg: 根据文档片段ID获取文档片段。如果文档片段不存在，则报错。
    segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_id).first()
    if not segment:
        raise NotFound("Segment not found")

    # cdg: 如果文档片段状态不是完成，则报错。
    if segment.status != "completed":
        raise NotFound("Segment is not completed, enable action is not allowed.")

    # cdg: 设置缓存键。
    indexing_cache_key = "segment_{}_indexing".format(segment.id)

    # cdg: 尝试启用文档片段索引。
    try:
        # cdg: 创建文档对象。
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            },
        )

        # cdg: 获取文档片段所属的知识库。
        dataset = segment.dataset

        # cdg: 如果文档片段所属的知识库不存在，则跳过。
        if not dataset:
            logging.info(click.style("Segment {} has no dataset, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 获取文档片段所属的文档。
        dataset_document = segment.document

        # cdg: 如果文档片段所属的文档不存在，则跳过。
        if not dataset_document:
            logging.info(click.style("Segment {} has no document, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 如果文档片段所属的文档状态不合法，则跳过。
        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
            logging.info(click.style("Segment {} document status is invalid, pass.".format(segment.id), fg="cyan"))
            return

        # cdg: 根据文档格式获取索引处理器。
        index_processor = IndexProcessorFactory(dataset_document.doc_form).init_index_processor()

        # cdg: 如果文档格式为父子索引，则处理子文档。
        if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
            child_chunks = segment.child_chunks
            if child_chunks:
                child_documents = []
                # cdg: 遍历子文档片段。
                for child_chunk in child_chunks:
                    # cdg: 创建子文档对象。
                    child_document = ChildDocument(
                        page_content=child_chunk.content,
                        metadata={
                            "doc_id": child_chunk.index_node_id,
                            "doc_hash": child_chunk.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                        },
                    )
                    # cdg: 添加子文档对象到列表。
                    child_documents.append(child_document)
                # cdg: 设置文档的子文档。
                document.children = child_documents

        # cdg: 保存向量索引。
        # save vector index
        index_processor.load(dataset, [document])

        end_at = time.perf_counter()
        logging.info(
            click.style("Segment enabled to index: {} latency: {}".format(segment.id, end_at - start_at), fg="green")
        )
    except Exception as e:
        logging.exception("enable segment to index failed")
        segment.enabled = False
        segment.disabled_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        segment.status = "error"
        segment.error = str(e)
        db.session.commit()
    finally:
        redis_client.delete(indexing_cache_key)
