import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset
from models.model import App, AppAnnotationSetting, MessageAnnotation
from services.dataset_service import DatasetCollectionBindingService

# cdg: 用于异步启用注释回复. 用法：celery -A celery_app.celery_app.celery worker -Q dataset -n worker1@%h
@shared_task(queue="dataset")
def enable_annotation_reply_task(
    job_id: str,
    app_id: str,
    user_id: str,
    tenant_id: str,
    score_threshold: float,
    embedding_provider_name: str,
    embedding_model_name: str,
):
    """
    Async enable annotation reply task
    """
    logging.info(click.style("Start add app annotation to index: {}".format(app_id), fg="green"))
    start_at = time.perf_counter()
    # get app info
    app = db.session.query(App).filter(App.id == app_id, App.tenant_id == tenant_id, App.status == "normal").first()

    if not app:
        raise NotFound("App not found")
    # cdg: 获取注释信息
    annotations = db.session.query(MessageAnnotation).filter(MessageAnnotation.app_id == app_id).all()
    enable_app_annotation_key = "enable_app_annotation_{}".format(str(app_id))
    enable_app_annotation_job_key = "enable_app_annotation_job_{}".format(str(job_id))

    try:
        # cdg: 初始化文档列表
        documents = []
        # cdg: 获取数据集集合绑定
        dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
            embedding_provider_name, embedding_model_name, "annotation"
        )
        # cdg: 获取应用注释设置
        annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )
        # cdg: 如果应用注释设置存在，则更新应用注释设置
        if annotation_setting:
            annotation_setting.score_threshold = score_threshold
            annotation_setting.collection_binding_id = dataset_collection_binding.id
            annotation_setting.updated_user_id = user_id
            annotation_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            db.session.add(annotation_setting)
        else:
            # cdg: 创建应用注释设置
            new_app_annotation_setting = AppAnnotationSetting(
                app_id=app_id,
                score_threshold=score_threshold,
                collection_binding_id=dataset_collection_binding.id,
                created_user_id=user_id,
                updated_user_id=user_id,
            )
            db.session.add(new_app_annotation_setting)

        # cdg: 创建数据集
        dataset = Dataset(
            id=app_id,
            tenant_id=tenant_id,
            indexing_technique="high_quality",
            embedding_model_provider=embedding_provider_name,
            embedding_model=embedding_model_name,
            collection_binding_id=dataset_collection_binding.id,
        )
        # cdg: 如果注释存在，则创建文档
        if annotations:
            # cdg: 创建文档
            for annotation in annotations:
                document = Document(
                    page_content=annotation.question,
                    metadata={"annotation_id": annotation.id, "app_id": app_id, "doc_id": annotation.id},
                )
                documents.append(document)

            # cdg: 创建向量索引工具
            vector = Vector(dataset, attributes=["doc_id", "annotation_id", "app_id"])
            try:
                # cdg: 删除注释索引
                vector.delete_by_metadata_field("app_id", app_id)
            except Exception as e:
                logging.info(click.style("Delete annotation index error: {}".format(str(e)), fg="red"))
            
            # cdg: 创建向量索引
            vector.create(documents)
        # cdg: 提交事务
        db.session.commit()
        # cdg: 设置启用注释缓存键
        redis_client.setex(enable_app_annotation_job_key, 600, "completed")
        # cdg: 计算执行时间
        end_at = time.perf_counter()
        logging.info(
            click.style("App annotations added to index: {} latency: {}".format(app_id, end_at - start_at), fg="green")
        )
    except Exception as e:
        logging.exception("Annotation batch created index failed")
        redis_client.setex(enable_app_annotation_job_key, 600, "error")
        enable_app_annotation_error_key = "enable_app_annotation_error_{}".format(str(job_id))
        redis_client.setex(enable_app_annotation_error_key, 600, str(e))
        db.session.rollback()
    finally:
        # cdg: 不管结果如何，都删除启用注释缓存键
        redis_client.delete(enable_app_annotation_key)
