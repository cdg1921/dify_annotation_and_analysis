import logging
import time

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.tools.utils.web_reader_tool import get_image_upload_file_ids
from extensions.ext_database import db
from extensions.ext_storage import storage
from models.dataset import Dataset, DocumentSegment
from models.model import UploadFile

# cdg: 异步清空文档数据。通过@shared_task装饰器，将batch_clean_document_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def batch_clean_document_task(document_ids: list[str], dataset_id: str, doc_form: str, file_ids: list[str]):
    """
    Clean document when document deleted.
    :param document_ids: document ids
    :param dataset_id: dataset id
    :param doc_form: doc_form
    :param file_ids: file ids

    Usage: clean_document_task.delay(document_id, dataset_id)
    """
    logging.info(click.style("Start batch clean documents when documents deleted", fg="green"))
    start_at = time.perf_counter()

    try:
        # cdg: 根据知识库ID获取知识库。
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

        # cdg: 如果知识库不存在，则抛出异常。
        if not dataset:
            raise Exception("Document has no dataset")

        # cdg: 根据文档ID获取文档的所有分段。
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id.in_(document_ids)).all()
        # check segment is exist
        if segments:
            # cdg: 获取所有分段的索引节点ID。
            index_node_ids = [segment.index_node_id for segment in segments]
            # cdg: 初始化索引处理器。
            index_processor = IndexProcessorFactory(doc_form).init_index_processor()
            # cdg: 清空知识库中所有文档数据。
            index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

            # cdg: 清空文档文件。
            for segment in segments:
                image_upload_file_ids = get_image_upload_file_ids(segment.content)
                for upload_file_id in image_upload_file_ids:
                    image_file = db.session.query(UploadFile).filter(UploadFile.id == upload_file_id).first()
                    try:
                        if image_file and image_file.key:
                            storage.delete(image_file.key)
                    except Exception:
                        logging.exception(
                            "Delete image_files failed when storage deleted, \
                                          image_upload_file_is: {}".format(upload_file_id)
                        )
                    db.session.delete(image_file)
                db.session.delete(segment)

            db.session.commit()

        # cdg: 进一步清空文件管理系统中的文档文件。
        if file_ids:
            files = db.session.query(UploadFile).filter(UploadFile.id.in_(file_ids)).all()
            for file in files:
                try:
                    storage.delete(file.key)
                except Exception:
                    logging.exception("Delete file failed when document deleted, file_id: {}".format(file.id))
                db.session.delete(file)
            db.session.commit()

        # cdg: 计算清空文档数据的时间，并记录日志。
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Cleaned documents when documents deleted latency: {}".format(end_at - start_at),
                fg="green",
            )
        )
    except Exception:
        logging.exception("Cleaned documents when documents deleted failed")
