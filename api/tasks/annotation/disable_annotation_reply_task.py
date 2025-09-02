import logging
import time

import click
from celery import shared_task  # type: ignore
from werkzeug.exceptions import NotFound

from core.rag.datasource.vdb.vector_factory import Vector
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset
from models.model import App, AppAnnotationSetting, MessageAnnotation

# cdg: 用于异步禁用注释回复. 用法：celery -A celery_app.celery_app.celery worker -Q dataset -n worker1@%h
@shared_task(queue="dataset")
def disable_annotation_reply_task(job_id: str, app_id: str, tenant_id: str):
    """
    Async enable annotation reply task
    """
    logging.info(click.style("Start delete app annotations index: {}".format(app_id), fg="green"))
    start_at = time.perf_counter()
    # get app info
    # cdg: 获取应用信息
    app = db.session.query(App).filter(App.id == app_id, App.tenant_id == tenant_id, App.status == "normal").first()

    # cdg: 获取注释数量
    annotations_count = db.session.query(MessageAnnotation).filter(MessageAnnotation.app_id == app_id).count()
    if not app:
        raise NotFound("App not found")
    # cdg: 获取应用注释设置
    app_annotation_setting = (
        db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
    )
    if not app_annotation_setting:
        raise NotFound("App annotation setting not found")
    # cdg: 拼接禁用注释缓存键
    disable_app_annotation_key = "disable_app_annotation_{}".format(str(app_id))
    disable_app_annotation_job_key = "disable_app_annotation_job_{}".format(str(job_id))

    # cdg: 创建数据集
    try:
        # cdg: 创建数据集
        dataset = Dataset(
            id=app_id,
            tenant_id=tenant_id,
            indexing_technique="high_quality",
            collection_binding_id=app_annotation_setting.collection_binding_id,
        )
        # cdg: 创建向量索引工具
        try:
            if annotations_count > 0:
                vector = Vector(dataset, attributes=["doc_id", "annotation_id", "app_id"])
                # cdg: 删除注释索引
                vector.delete_by_metadata_field("app_id", app_id)
        except Exception:
            logging.exception("Delete annotation index failed when annotation deleted.")
        # cdg: 设置禁用注释缓存键
        redis_client.setex(disable_app_annotation_job_key, 600, "completed")

        # cdg: 删除应用注释设置
        db.session.delete(app_annotation_setting)
        db.session.commit()

        # cdg: 计算执行时间
        end_at = time.perf_counter()
        logging.info(
            click.style("App annotations index deleted : {} latency: {}".format(app_id, end_at - start_at), fg="green")
        )
    except Exception as e:
        # cdg: 记录异常
        logging.exception("Annotation batch deleted index failed")
        redis_client.setex(disable_app_annotation_job_key, 600, "error")
        disable_app_annotation_error_key = "disable_app_annotation_error_{}".format(str(job_id))
        redis_client.setex(disable_app_annotation_error_key, 600, str(e))
    finally:
        # cdg: 不管结果如何，都删除禁用注释缓存键
        redis_client.delete(disable_app_annotation_key)
