import datetime
import uuid
from typing import cast

import pandas as pd
from flask_login import current_user  # type: ignore
from sqlalchemy import or_
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import NotFound

from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.model import App, AppAnnotationHitHistory, AppAnnotationSetting, Message, MessageAnnotation
from services.feature_service import FeatureService
from tasks.annotation.add_annotation_to_index_task import add_annotation_to_index_task
from tasks.annotation.batch_import_annotations_task import batch_import_annotations_task
from tasks.annotation.delete_annotation_index_task import delete_annotation_index_task
from tasks.annotation.disable_annotation_reply_task import disable_annotation_reply_task
from tasks.annotation.enable_annotation_reply_task import enable_annotation_reply_task
from tasks.annotation.update_annotation_to_index_task import update_annotation_to_index_task

# cdg: 应用注释服务，包括插入应用注释、启用应用注释、禁用应用注释、获取应用注释列表、导出应用注释列表、直接插入应用注释、更新应用注释、删除应用注释、批量导入应用注释、获取应用注释命中历史、获取应用注释、添加应用注释历史、获取应用注释设置、更新应用注释设置。
class AppAnnotationService:
    # cdg: 插入应用注释，具体实现为：获取应用信息；如果消息ID存在，则获取消息信息；如果消息存在，则获取消息注释；如果消息注释存在，则更新消息注释；否则创建消息注释；如果注释回复启用，则添加注释到索引。
    @classmethod
    def up_insert_app_annotation_from_message(cls, args: dict, app_id: str) -> MessageAnnotation:
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")
        if args.get("message_id"):
            message_id = str(args["message_id"])
            # get message info
            message = db.session.query(Message).filter(Message.id == message_id, Message.app_id == app.id).first()

            if not message:
                raise NotFound("Message Not Exists.")

            annotation = message.annotation
            # save the message annotation
            if annotation:
                annotation.content = args["answer"]
                annotation.question = args["question"]
            else:
                annotation = MessageAnnotation(
                    app_id=app.id,
                    conversation_id=message.conversation_id,
                    message_id=message.id,
                    content=args["answer"],
                    question=args["question"],
                    account_id=current_user.id,
                )
        else:
            annotation = MessageAnnotation(
                app_id=app.id, content=args["answer"], question=args["question"], account_id=current_user.id
            )
        db.session.add(annotation)
        db.session.commit()
        # if annotation reply is enabled , add annotation to index
        annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )
        if annotation_setting:
            add_annotation_to_index_task.delay(
                annotation.id,
                args["question"],
                current_user.current_tenant_id,
                app_id,
                annotation_setting.collection_binding_id,
            )
        return cast(MessageAnnotation, annotation)

    # cdg: 启用应用注释，具体实现为：如果应用注释启用缓存存在，则返回缓存结果；否则异步添加注释到索引。
    @classmethod
    def enable_app_annotation(cls, args: dict, app_id: str) -> dict:
        enable_app_annotation_key = "enable_app_annotation_{}".format(str(app_id))
        cache_result = redis_client.get(enable_app_annotation_key)
        if cache_result is not None:
            return {"job_id": cache_result, "job_status": "processing"}

        # async job
        job_id = str(uuid.uuid4())
        enable_app_annotation_job_key = "enable_app_annotation_job_{}".format(str(job_id))
        # send batch add segments task
        redis_client.setnx(enable_app_annotation_job_key, "waiting")
        enable_annotation_reply_task.delay(
            str(job_id),
            app_id,
            current_user.id,
            current_user.current_tenant_id,
            args["score_threshold"],
            args["embedding_provider_name"],
            args["embedding_model_name"],
        )
        return {"job_id": job_id, "job_status": "waiting"}

    # cdg: 禁用应用注释，具体实现为：如果应用注释禁用缓存存在，则返回缓存结果；否则异步添加注释到索引。
    @classmethod
    def disable_app_annotation(cls, app_id: str) -> dict:
        disable_app_annotation_key = "disable_app_annotation_{}".format(str(app_id))
        cache_result = redis_client.get(disable_app_annotation_key)
        if cache_result is not None:
            return {"job_id": cache_result, "job_status": "processing"}

        # async job
        job_id = str(uuid.uuid4())
        disable_app_annotation_job_key = "disable_app_annotation_job_{}".format(str(job_id))
        # send batch add segments task
        redis_client.setnx(disable_app_annotation_job_key, "waiting")
        disable_annotation_reply_task.delay(str(job_id), app_id, current_user.current_tenant_id)
        return {"job_id": job_id, "job_status": "waiting"}

    # cdg: 获取应用注释列表，具体实现为：获取应用信息；如果应用不存在，则抛出异常；如果关键字存在，则获取应用注释列表；否则获取应用注释列表。
    @classmethod
    def get_annotation_list_by_app_id(cls, app_id: str, page: int, limit: int, keyword: str):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")
        if keyword:
            annotations = (
                MessageAnnotation.query.filter(MessageAnnotation.app_id == app_id)
                .filter(
                    or_(
                        MessageAnnotation.question.ilike("%{}%".format(keyword)),
                        MessageAnnotation.content.ilike("%{}%".format(keyword)),
                    )
                )
                .order_by(MessageAnnotation.created_at.desc(), MessageAnnotation.id.desc())
                .paginate(page=page, per_page=limit, max_per_page=100, error_out=False)
            )
        else:
            annotations = (
                MessageAnnotation.query.filter(MessageAnnotation.app_id == app_id)
                .order_by(MessageAnnotation.created_at.desc(), MessageAnnotation.id.desc())
                .paginate(page=page, per_page=limit, max_per_page=100, error_out=False)
            )
        return annotations.items, annotations.total

    # cdg: 导出应用注释列表，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释列表。
    @classmethod
    def export_annotation_list_by_app_id(cls, app_id: str):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")
        annotations = (
            db.session.query(MessageAnnotation)
            .filter(MessageAnnotation.app_id == app_id)
            .order_by(MessageAnnotation.created_at.desc())
            .all()
        )
        return annotations

    # cdg: 直接插入应用注释，具体实现为：获取应用信息；如果应用不存在，则抛出异常；创建应用注释；如果注释回复启用，则添加注释到索引。
    @classmethod
    def insert_app_annotation_directly(cls, args: dict, app_id: str) -> MessageAnnotation:
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation = MessageAnnotation(
            app_id=app.id, content=args["answer"], question=args["question"], account_id=current_user.id
        )
        db.session.add(annotation)
        db.session.commit()
        # if annotation reply is enabled , add annotation to index
        annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )
        if annotation_setting:
            add_annotation_to_index_task.delay(
                annotation.id,
                args["question"],
                current_user.current_tenant_id,
                app_id,
                annotation_setting.collection_binding_id,
            )
        return annotation

    # cdg: 更新应用注释，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释；如果应用注释不存在，则抛出异常；更新应用注释；如果注释回复启用，则添加注释到索引。
    @classmethod
    def update_app_annotation_directly(cls, args: dict, app_id: str, annotation_id: str):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation = db.session.query(MessageAnnotation).filter(MessageAnnotation.id == annotation_id).first()

        if not annotation:
            raise NotFound("Annotation not found")

        annotation.content = args["answer"]
        annotation.question = args["question"]

        db.session.commit()
        # if annotation reply is enabled , add annotation to index
        app_annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )

        if app_annotation_setting:
            update_annotation_to_index_task.delay(
                annotation.id,
                annotation.question,
                current_user.current_tenant_id,
                app_id,
                app_annotation_setting.collection_binding_id,
            )

        return annotation

    # cdg: 删除应用注释，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释；如果应用注释不存在，则抛出异常；删除应用注释；删除应用注释命中历史；如果注释回复启用，则删除注释索引。
    @classmethod
    def delete_app_annotation(cls, app_id: str, annotation_id: str):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation = db.session.query(MessageAnnotation).filter(MessageAnnotation.id == annotation_id).first()

        if not annotation:
            raise NotFound("Annotation not found")

        db.session.delete(annotation)

        annotation_hit_histories = (
            db.session.query(AppAnnotationHitHistory)
            .filter(AppAnnotationHitHistory.annotation_id == annotation_id)
            .all()
        )
        if annotation_hit_histories:
            for annotation_hit_history in annotation_hit_histories:
                db.session.delete(annotation_hit_history)

        db.session.commit()
        # if annotation reply is enabled , delete annotation index
        app_annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )

        if app_annotation_setting:
            delete_annotation_index_task.delay(
                annotation.id, app_id, current_user.current_tenant_id, app_annotation_setting.collection_binding_id
            )

    # cdg: 批量导入应用注释，具体实现为：获取应用信息；如果应用不存在，则抛出异常；读取CSV文件；如果CSV文件为空，则抛出异常；检查注释限制；异步添加注释到索引。
    @classmethod
    def batch_import_app_annotations(cls, app_id, file: FileStorage) -> dict:
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        try:
            # Skip the first row
            df = pd.read_csv(file)
            result = []
            for index, row in df.iterrows():
                content = {"question": row.iloc[0], "answer": row.iloc[1]}
                result.append(content)
            if len(result) == 0:
                raise ValueError("The CSV file is empty.")
            # check annotation limit
            features = FeatureService.get_features(current_user.current_tenant_id)
            if features.billing.enabled:
                annotation_quota_limit = features.annotation_quota_limit
                if annotation_quota_limit.limit < len(result) + annotation_quota_limit.size:
                    raise ValueError("The number of annotations exceeds the limit of your subscription.")
            # async job
            job_id = str(uuid.uuid4())
            indexing_cache_key = "app_annotation_batch_import_{}".format(str(job_id))
            # send batch add segments task
            redis_client.setnx(indexing_cache_key, "waiting")
            batch_import_annotations_task.delay(
                str(job_id), result, app_id, current_user.current_tenant_id, current_user.id
            )
        except Exception as e:
            return {"error_msg": str(e)}
        return {"job_id": job_id, "job_status": "waiting"}

    # cdg: 获取应用注释命中历史，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释；如果应用注释不存在，则抛出异常；获取应用注释命中历史。
    @classmethod
    def get_annotation_hit_histories(cls, app_id: str, annotation_id: str, page, limit):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation = db.session.query(MessageAnnotation).filter(MessageAnnotation.id == annotation_id).first()

        if not annotation:
            raise NotFound("Annotation not found")

        annotation_hit_histories = (
            AppAnnotationHitHistory.query.filter(
                AppAnnotationHitHistory.app_id == app_id,
                AppAnnotationHitHistory.annotation_id == annotation_id,
            )
            .order_by(AppAnnotationHitHistory.created_at.desc())
            .paginate(page=page, per_page=limit, max_per_page=100, error_out=False)
        )
        return annotation_hit_histories.items, annotation_hit_histories.total

    # cdg: 获取应用注释，具体实现为：获取应用注释；如果应用注释不存在，则返回None。
    @classmethod
    def get_annotation_by_id(cls, annotation_id: str) -> MessageAnnotation | None:
        annotation = db.session.query(MessageAnnotation).filter(MessageAnnotation.id == annotation_id).first()

        if not annotation:
            return None
        return annotation

    # cdg: 添加应用注释历史，具体实现为：获取应用注释；如果应用注释不存在，则返回None。
    @classmethod
    def add_annotation_history(
        cls,
        annotation_id: str,
        app_id: str,
        annotation_question: str,
        annotation_content: str,
        query: str,
        user_id: str,
        message_id: str,
        from_source: str,
        score: float,
    ):
        # add hit count to annotation
        db.session.query(MessageAnnotation).filter(MessageAnnotation.id == annotation_id).update(
            {MessageAnnotation.hit_count: MessageAnnotation.hit_count + 1}, synchronize_session=False
        )

        annotation_hit_history = AppAnnotationHitHistory(
            annotation_id=annotation_id,
            app_id=app_id,
            account_id=user_id,
            question=query,
            source=from_source,
            score=score,
            message_id=message_id,
            annotation_question=annotation_question,
            annotation_content=annotation_content,
        )
        db.session.add(annotation_hit_history)
        db.session.commit()

    # cdg: 根据应用ID获取应用注释设置，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释设置；如果应用注释设置存在，则获取应用注释设置的集合绑定详情；返回应用注释设置。
    @classmethod
    def get_app_annotation_setting_by_app_id(cls, app_id: str):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation_setting = (
            db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.app_id == app_id).first()
        )
        if annotation_setting:
            collection_binding_detail = annotation_setting.collection_binding_detail
            return {
                "id": annotation_setting.id,
                "enabled": True,
                "score_threshold": annotation_setting.score_threshold,
                "embedding_model": {
                    "embedding_provider_name": collection_binding_detail.provider_name,
                    "embedding_model_name": collection_binding_detail.model_name,
                },
            }
        return {"enabled": False}

    # cdg: 更新应用注释设置，具体实现为：获取应用信息；如果应用不存在，则抛出异常；获取应用注释设置；如果应用注释设置不存在，则抛出异常；更新应用注释设置；获取应用注释设置的集合绑定详情；返回应用注释设置。
    @classmethod
    def update_app_annotation_setting(cls, app_id: str, annotation_setting_id: str, args: dict):
        # get app info
        app = (
            db.session.query(App)
            .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
            .first()
        )

        if not app:
            raise NotFound("App not found")

        annotation_setting = (
            db.session.query(AppAnnotationSetting)
            .filter(
                AppAnnotationSetting.app_id == app_id,
                AppAnnotationSetting.id == annotation_setting_id,
            )
            .first()
        )
        if not annotation_setting:
            raise NotFound("App annotation not found")
        annotation_setting.score_threshold = args["score_threshold"]
        annotation_setting.updated_user_id = current_user.id
        annotation_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        db.session.add(annotation_setting)
        db.session.commit()

        collection_binding_detail = annotation_setting.collection_binding_detail

        return {
            "id": annotation_setting.id,
            "enabled": True,
            "score_threshold": annotation_setting.score_threshold,
            "embedding_model": {
                "embedding_provider_name": collection_binding_detail.provider_name,
                "embedding_model_name": collection_binding_detail.model_name,
            },
        }



# cdg: 这份代码是应用注释服务，包括插入应用注释、启用应用注释、禁用应用注释、获取应用注释列表、导出应用注释列表、直接插入应用注释、更新应用注释、删除应用注释、批量导入应用注释、获取应用注释命中历史、获取应用注释、添加应用注释历史、获取应用注释设置、更新应用注释设置。
# cdg: 其中用到的一些关键技术点：
# cdg: 1. 使用Flask-Login获取当前用户信息。
# cdg: 2. 使用SQLAlchemy进行数据库操作。
# cdg: 3. 使用Redis进行缓存。
# cdg: 4. 使用Celery进行异步任务。
# cdg: 5. 使用Pandas进行CSV文件操作。
# cdg: 6. 使用Werkzeug进行文件上传。
# cdg: 7. 使用Werkzeug进行异常处理。
# cdg: 8. 使用Werkzeug进行HTTP请求。
# cdg: 9. 使用Werkzeug进行HTTP响应。