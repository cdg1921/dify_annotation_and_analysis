import logging
import time

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.constant.index_type import IndexType
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import ChildDocument, Document
from extensions.ext_database import db
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument

# cdg: 异步处理知识库的向量索引。通过@shared_task装饰器，将deal_dataset_vector_index_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def deal_dataset_vector_index_task(dataset_id: str, action: str):
    """
    Async deal dataset from index
    :param dataset_id: dataset_id
    :param action: action
    Usage: deal_dataset_vector_index_task.delay(dataset_id, action)
    """
    logging.info(click.style("Start deal dataset vector index: {}".format(dataset_id), fg="green"))
    start_at = time.perf_counter()

    try:
        # cdg: 根据知识库ID获取知识库。
        dataset = Dataset.query.filter_by(id=dataset_id).first()
        # cdg: 如果知识库不存在，则抛出异常。
        if not dataset:
            raise Exception("Dataset not found")

        # cdg: 获取知识库的索引类型。
        index_type = dataset.doc_form or IndexType.PARAGRAPH_INDEX
        # cdg: 初始化索引处理器。
        index_processor = IndexProcessorFactory(index_type).init_index_processor()

        # cdg: 如果操作类型为删除，则清空知识库中的所有文档。
        if action == "remove":
            index_processor.clean(dataset, None, with_keywords=False)
        elif action == "add":  # cdg: 如果操作类型为添加，则添加知识库中的所有文档。
            # cdg: 获取知识库中的所有文档。
            dataset_documents = (
                db.session.query(DatasetDocument)
                .filter(
                    DatasetDocument.dataset_id == dataset_id,
                    DatasetDocument.indexing_status == "completed",
                    DatasetDocument.enabled == True,
                    DatasetDocument.archived == False,
                )
                .all()
            )
            # cdg: 如果文档存在，则更新文档的索引状态。
            if dataset_documents:
                # cdg: 获取文档的ID。
                dataset_documents_ids = [doc.id for doc in dataset_documents]
                db.session.query(DatasetDocument).filter(DatasetDocument.id.in_(dataset_documents_ids)).update(
                    {"indexing_status": "indexing"}, synchronize_session=False
                )
                db.session.commit()

                # cdg: 遍历文档。
                for dataset_document in dataset_documents:
                    try:
                        # cdg: 获取文档中的所有分段。   
                        # add from vector index
                        segments = (
                            db.session.query(DocumentSegment)
                            .filter(DocumentSegment.document_id == dataset_document.id, DocumentSegment.enabled == True)
                            .order_by(DocumentSegment.position.asc())
                            .all()
                        )
                        # cdg: 如果分段存在，则构建Segment对象。
                        if segments:
                            documents = []
                            # cdg: 遍历分段。   
                            for segment in segments:
                                document = Document(
                                    page_content=segment.content,
                                    metadata={
                                        "doc_id": segment.index_node_id,
                                        "doc_hash": segment.index_node_hash,
                                        "document_id": segment.document_id,
                                        "dataset_id": segment.dataset_id,
                                    },
                                )
                                # cdg: 将segment对象添加到列表中。
                                documents.append(document)
                            # save vector index
                            # cdg: 将文档添加到向量库中。
                            index_processor.load(dataset, documents, with_keywords=False)
                        # cdg: 更新文档的索引状态。
                        db.session.query(DatasetDocument).filter(DatasetDocument.id == dataset_document.id).update(
                            {"indexing_status": "completed"}, synchronize_session=False
                        )
                        db.session.commit()
                    except Exception as e:
                        db.session.query(DatasetDocument).filter(DatasetDocument.id == dataset_document.id).update(
                            {"indexing_status": "error", "error": str(e)}, synchronize_session=False
                        )
                        db.session.commit()
        elif action == "update": # cdg: 如果操作类型为更新，则更新知识库中的所有文档。
            # cdg: 获取知识库中的所有文档。
            dataset_documents = (
                db.session.query(DatasetDocument)
                .filter(
                    DatasetDocument.dataset_id == dataset_id,
                    DatasetDocument.indexing_status == "completed",
                    DatasetDocument.enabled == True,
                    DatasetDocument.archived == False,
                )
                .all()
            )
            # cdg: 如果文档存在，则更新文档的索引状态。
            # add new index
            if dataset_documents:
                # cdg: 获取文档的ID。
                dataset_documents_ids = [doc.id for doc in dataset_documents]
                db.session.query(DatasetDocument).filter(DatasetDocument.id.in_(dataset_documents_ids)).update(
                    {"indexing_status": "indexing"}, synchronize_session=False
                )
                db.session.commit()

                # cdg: 清空知识库中的所有文档。
                index_processor.clean(dataset, None, with_keywords=False, delete_child_chunks=False)

                # cdg: 遍历文档。
                for dataset_document in dataset_documents:
                    # update from vector index
                    try:
                        segments = (
                            db.session.query(DocumentSegment)
                            .filter(DocumentSegment.document_id == dataset_document.id, DocumentSegment.enabled == True)
                            .order_by(DocumentSegment.position.asc())
                            .all()
                        )
                        if segments:
                            documents = []
                            for segment in segments:
                                document = Document(
                                    page_content=segment.content,
                                    metadata={
                                        "doc_id": segment.index_node_id,
                                        "doc_hash": segment.index_node_hash,
                                        "document_id": segment.document_id,
                                        "dataset_id": segment.dataset_id,
                                    },
                                )
                                if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
                                    child_chunks = segment.child_chunks
                                    if child_chunks:
                                        child_documents = []
                                        for child_chunk in child_chunks:
                                            child_document = ChildDocument(
                                                page_content=child_chunk.content,
                                                metadata={
                                                    "doc_id": child_chunk.index_node_id,
                                                    "doc_hash": child_chunk.index_node_hash,
                                                    "document_id": segment.document_id,
                                                    "dataset_id": segment.dataset_id,
                                                },
                                            )
                                            child_documents.append(child_document)
                                        document.children = child_documents
                                documents.append(document)
                            # save vector index
                            index_processor.load(dataset, documents, with_keywords=False)
                        db.session.query(DatasetDocument).filter(DatasetDocument.id == dataset_document.id).update(
                            {"indexing_status": "completed"}, synchronize_session=False
                        )
                        db.session.commit()
                    except Exception as e:
                        db.session.query(DatasetDocument).filter(DatasetDocument.id == dataset_document.id).update(
                            {"indexing_status": "error", "error": str(e)}, synchronize_session=False
                        )
                        db.session.commit()
            else: # cdg: 其他情况，则清空知识库中的所有文档。
                # clean collection
                index_processor.clean(dataset, None, with_keywords=False, delete_child_chunks=False)

        end_at = time.perf_counter()
        logging.info(
            click.style("Deal dataset vector index: {} latency: {}".format(dataset_id, end_at - start_at), fg="green")
        )
    except Exception:
        logging.exception("Deal dataset vector index failed")
