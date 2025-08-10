import logging
import time
from typing import Optional

import click
from celery import shared_task  # type: ignore

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.tools.utils.web_reader_tool import get_image_upload_file_ids
from extensions.ext_database import db
from extensions.ext_storage import storage
from models.dataset import Dataset, DocumentSegment
from models.model import UploadFile

# cdg: 异步清空文档。通过@shared_task装饰器，将clean_document_task函数注册为Celery任务，并指定任务队列为dataset。
@shared_task(queue="dataset")
def clean_document_task(document_id: str, dataset_id: str, doc_form: str, file_id: Optional[str]):
    """
    Clean document when document deleted.
    :param document_id: document id
    :param dataset_id: dataset id
    :param doc_form: doc_form
    :param file_id: file id

    Usage: clean_document_task.delay(document_id, dataset_id)
    """
    logging.info(click.style("Start clean document when document deleted: {}".format(document_id), fg="green"))
    start_at = time.perf_counter()

    try:
        # cdg: 根据文档ID获取文档所属的知识库。
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        # cdg: 如果文档没有所属的知识库，则抛出异常。
        if not dataset:
            raise Exception("Document has no dataset")

        # cdg: 根据文档ID获取文档的所有分段。
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
        # check segment is exist
        if segments:
            # cdg: 获取所有分段的索引节点ID。
            index_node_ids = [segment.index_node_id for segment in segments]
            # cdg: 初始化索引处理器。
            index_processor = IndexProcessorFactory(doc_form).init_index_processor()
            # cdg: 清空文档中的所有分段。
            index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

            # cdg: 遍历文档中的所有分段。
            for segment in segments:
                # cdg: 获取分段中的所有图片文件ID。
                image_upload_file_ids = get_image_upload_file_ids(segment.content)
                for upload_file_id in image_upload_file_ids:
                    image_file = db.session.query(UploadFile).filter(UploadFile.id == upload_file_id).first()
                    # cdg: 如果图片文件不存在，则跳过。
                    if image_file is None:
                        continue
                    # cdg: 删除图片文件。
                    try:
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
        if file_id:
            # cdg: 根据文件ID获取文件。
            file = db.session.query(UploadFile).filter(UploadFile.id == file_id).first()
            # cdg: 如果文件不存在，则跳过。
            if file:
                try:
                    # cdg: 删除文件。
                    storage.delete(file.key)
                except Exception:
                    logging.exception("Delete file failed when document deleted, file_id: {}".format(file_id))
                db.session.delete(file)
                db.session.commit()

        # cdg: 计算清空文档的时间，并记录日志。
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Cleaned document when document deleted: {} latency: {}".format(document_id, end_at - start_at),
                fg="green",
            )
        )
    except Exception:
        logging.exception("Cleaned document when document deleted failed")
