import datetime
import logging
import time
import uuid

import click
from celery import shared_task  # type: ignore
from sqlalchemy import func

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs import helper
from models.dataset import Dataset, Document, DocumentSegment
from services.vector_service import VectorService

# cdg: 异步批量创建文本分段并添加到向量库中。通过@shared_task装饰器，将batch_create_segment_to_index_task函数注册为Celery任务，并指定任务队列为dataset。
# cdg: 这个任务的主要作用是实现将文档中的文本分段批量添加到向量库中。
@shared_task(queue="dataset")
def batch_create_segment_to_index_task(
    job_id: str, content: list, dataset_id: str, document_id: str, tenant_id: str, user_id: str
):
    """
    Async batch create segment to index
    :param job_id:
    :param content:
    :param dataset_id:
    :param document_id:
    :param tenant_id:
    :param user_id:

    Usage: batch_create_segment_to_index_task.delay(segment_id)
    """
    logging.info(click.style("Start batch create segment jobId: {}".format(job_id), fg="green"))
    start_at = time.perf_counter()

    indexing_cache_key = "segment_batch_import_{}".format(job_id)

    try:
        # cdg: 根据知识库ID获取知识库。
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        # cdg: 如果知识库不存在，则抛出异常。
        if not dataset:
            raise ValueError("Dataset not exist.")

        # cdg: 根据文档ID获取文档。
        dataset_document = db.session.query(Document).filter(Document.id == document_id).first()
        # cdg: 如果文档不存在，则抛出异常。
        if not dataset_document:
            raise ValueError("Document not exist.")

        # cdg: 如果文档不可用，则抛出异常。
        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != "completed":
            raise ValueError("Document is not available.")
        
        # cdg: 初始化文本分段列表。
        document_segments = []
        embedding_model = None
        if dataset.indexing_technique == "high_quality":
            # cdg: 高质量索引模式下，需要使用模型计算文本分段的embedding。
            # cdg: 初始化模型管理器。
            model_manager = ModelManager()
            # cdg: 获取Embedding模型实例。
            embedding_model = model_manager.get_model_instance(
                tenant_id=dataset.tenant_id,
                provider=dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model,
            )

        # cdg: 初始化文本分段列表。
        word_count_change = 0
        segments_to_insert: list[str] = []  # Explicitly type hint the list as List[str]
        # cdg: 遍历文本分段列表。
        for segment in content:
            # cdg: 获取文本分段的内容。
            content_str = segment["content"]
            # cdg: 生成文本分段的唯一ID。
            doc_id = str(uuid.uuid4())
            # cdg: 生成文本分段的哈希值。
            segment_hash = helper.generate_text_hash(content_str)
            # cdg: 计算文本分段的token数量。
            # calc embedding use tokens
            tokens = embedding_model.get_text_embedding_num_tokens(texts=[content_str]) if embedding_model else 0
            # cdg: 获取文档中最后一个文本分段的位置。
            max_position = (
                db.session.query(func.max(DocumentSegment.position))
                .filter(DocumentSegment.document_id == dataset_document.id)
                .scalar()
            )
            # cdg: 创建文本分段DocumentSegment对象。
            segment_document = DocumentSegment(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                index_node_id=doc_id,
                index_node_hash=segment_hash,
                position=max_position + 1 if max_position else 1,
                content=content_str,
                word_count=len(content_str),
                tokens=tokens,
                created_by=user_id,
                indexing_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
                status="completed",
                completed_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
            )
            # cdg: 如果文档的索引方式为问答模型，则设置文本分段的答案。
            if dataset_document.doc_form == "qa_model":
                segment_document.answer = segment["answer"]
                segment_document.word_count += len(segment["answer"])
            word_count_change += segment_document.word_count
            db.session.add(segment_document)
            document_segments.append(segment_document)
            segments_to_insert.append(str(segment))  # Cast to string if needed
        # cdg: 更新文档的长度。
        # update document word count
        dataset_document.word_count += word_count_change
        db.session.add(dataset_document)

        # cdg: 将文本分段添加到向量库中。
        # add index to db
        VectorService.create_segments_vector(None, document_segments, dataset, dataset_document.doc_form)
        # cdg: 提交事务。
        db.session.commit()
        # cdg: 设置Redis中的索引缓存键。
        redis_client.setex(indexing_cache_key, 600, "completed")
        end_at = time.perf_counter()
        logging.info(
            click.style("Segment batch created job: {} latency: {}".format(job_id, end_at - start_at), fg="green")
        )
    except Exception as e:
        logging.exception("Segments batch created index failed")
        redis_client.setex(indexing_cache_key, 600, "error")
