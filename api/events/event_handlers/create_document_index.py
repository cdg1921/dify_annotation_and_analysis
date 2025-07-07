import datetime
import logging
import time

import click
from werkzeug.exceptions import NotFound

from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from events.event_handlers.document_index_event import document_index_created
from extensions.ext_database import db
from models.dataset import Document

# cdg: 使用@document_index_created.connect装饰器，将handle函数注册为文档索引创建事件的处理器。
@document_index_created.connect
def handle(sender, **kwargs): # cdg: 处理文档索引创建事件，sender是文档对象，kwargs是事件参数   
    dataset_id = sender # cdg: 获取数据集ID
    document_ids = kwargs.get("document_ids", []) # cdg: 获取文档ID列表
    documents = [] # cdg: 创建文档列表
    start_at = time.perf_counter() # cdg: 获取开始时间
    # cdg: 遍历文档ID列表，创建文档索引,将文档信息保存到数据库中
    for document_id in document_ids:
        logging.info(click.style("Start process document: {}".format(document_id), fg="green"))

        document = (
            db.session.query(Document)
            .filter(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
            )
            .first()
        )

        if not document:
            raise NotFound("Document not found")

        document.indexing_status = "parsing"
        document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        documents.append(document)
        db.session.add(document)
    db.session.commit() # cdg: 提交文档信息保存事务

    try:
        indexing_runner = IndexingRunner() # cdg: 创建索引运行器
        indexing_runner.run(documents) # cdg: 文档处理，包括文档切分、向量化、索引等 
        end_at = time.perf_counter() # cdg: 获取结束时间
        logging.info(click.style("Processed dataset: {} latency: {}".format(dataset_id, end_at - start_at), fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        pass
