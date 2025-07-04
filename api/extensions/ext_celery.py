from datetime import timedelta

import pytz
from celery import Celery, Task  # type: ignore
from celery.schedules import crontab  # type: ignore

from configs import dify_config
from dify_app import DifyApp

# cdg: 初始化Celery
def init_app(app: DifyApp) -> Celery:
    # cdg: 创建FlaskTask类，继承自Task类，用于在Celery中使用Flask的上下文  
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context(): #cdg: 使用Flask的上下文
                return self.run(*args, **kwargs) #cdg: 运行任务

    broker_transport_options = {} #cdg: 设置broker_transport_options

    # cdg: 使用Sentinel。Sentinel是Redis的哨兵机制，用于监控Redis主节点的健康状态，并在主节点故障时自动切换到备用节点。
    if dify_config.CELERY_USE_SENTINEL:
        broker_transport_options = {    
            "master_name": dify_config.CELERY_SENTINEL_MASTER_NAME, #cdg: 设置主节点名称
            "sentinel_kwargs": {
                "socket_timeout": dify_config.CELERY_SENTINEL_SOCKET_TIMEOUT, #cdg: 设置哨兵超时时间
            },
        }

    # cdg: 初始化Celery
    celery_app = Celery(
        app.name, #cdg: 设置应用名称
        task_cls=FlaskTask, #cdg: 设置任务类
        broker=dify_config.CELERY_BROKER_URL, #cdg: 设置broker URL
        backend=dify_config.CELERY_BACKEND, #cdg: 设置backend
        task_ignore_result=True, #cdg: 设置忽略结果
    )

    # cdg: 添加SSL选项到Celery配置
    ssl_options = {
        "ssl_cert_reqs": None, #cdg: 设置证书请求为None
        "ssl_ca_certs": None, #cdg: 设置CA证书
        "ssl_certfile": None, #cdg: 设置证书文件
        "ssl_keyfile": None, #cdg: 设置密钥文件
    }

    # cdg: 更新Celery配置
    celery_app.conf.update(
        result_backend=dify_config.CELERY_RESULT_BACKEND, #cdg: 设置结果backend
        broker_transport_options=broker_transport_options, #cdg: 设置broker transport options
        broker_connection_retry_on_startup=True, #cdg: 设置连接重试
        worker_log_format=dify_config.LOG_FORMAT, #cdg: 设置日志格式
        worker_task_log_format=dify_config.LOG_FORMAT, #cdg: 设置任务日志格式
        worker_hijack_root_logger=False, #cdg: 设置劫持根日志
        timezone=pytz.timezone(dify_config.LOG_TZ or "UTC"), #cdg: 设置时区
    )

    if dify_config.BROKER_USE_SSL:
        celery_app.conf.update(
            broker_use_ssl=ssl_options,  # cdg: 添加SSL选项到broker配置
        )

    # cdg: 更新Celery配置
    if dify_config.LOG_FILE:
        celery_app.conf.update(
            worker_logfile=dify_config.LOG_FILE,
        )

    # cdg: 设置默认配置
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    # cdg: 导入任务
    imports = [
        "schedule.clean_embedding_cache_task", #cdg: 导入清理嵌入缓存任务
        "schedule.clean_unused_datasets_task", #cdg: 导入清理未使用数据集任务
        "schedule.create_tidb_serverless_task", #cdg: 导入创建TiDB Serverless任务
        "schedule.update_tidb_serverless_status_task", #cdg: 导入更新TiDB Serverless状态任务
        "schedule.clean_messages", #cdg: 导入清理消息任务
        "schedule.mail_clean_document_notify_task", #cdg: 导入清理文档通知任务
    ]
    day = dify_config.CELERY_BEAT_SCHEDULER_TIME #cdg: 设置beat_schedule时间
    beat_schedule = {
        "clean_embedding_cache_task": { #cdg: 设置清理嵌入缓存任务
            "task": "schedule.clean_embedding_cache_task.clean_embedding_cache_task", #cdg: 设置任务
            "schedule": timedelta(days=day), #cdg: 设置时间，每天执行
        },
        "clean_unused_datasets_task": { #cdg: 设置清理未使用数据集任务
            "task": "schedule.clean_unused_datasets_task.clean_unused_datasets_task", #cdg: 设置任务
            "schedule": timedelta(days=day), #cdg: 设置时间，每天执行
        },
        "create_tidb_serverless_task": { #cdg: 设置创建TiDB Serverless任务
            "task": "schedule.create_tidb_serverless_task.create_tidb_serverless_task", #cdg: 设置任务
            "schedule": crontab(minute="0", hour="*"), #cdg: 设置时间，每小时执行
        },
        "update_tidb_serverless_status_task": { #cdg: 设置更新TiDB Serverless状态任务
            "task": "schedule.update_tidb_serverless_status_task.update_tidb_serverless_status_task", #cdg: 设置任务
            "schedule": timedelta(minutes=10), #cdg: 设置时间，每10分钟执行
        },
        "clean_messages": { #cdg: 设置清理消息任务
            "task": "schedule.clean_messages.clean_messages", #cdg: 设置任务
            "schedule": timedelta(days=day), #cdg: 设置时间，每天执行
        },
        # cdg: 每周一执行
        "mail_clean_document_notify_task": { #cdg: 设置清理文档通知任务
            "task": "schedule.mail_clean_document_notify_task.mail_clean_document_notify_task", #cdg: 设置任务
            "schedule": crontab(minute="0", hour="10", day_of_week="1"), #cdg: 设置时间，每周一10点执行
        },
    }
    celery_app.conf.update(beat_schedule=beat_schedule, imports=imports) #cdg: 更新beat_schedule，beat_schedule即心跳调度任务

    return celery_app
