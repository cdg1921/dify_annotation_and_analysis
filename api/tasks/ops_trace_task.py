import json
import logging

from celery import shared_task  # type: ignore
from flask import current_app

from core.ops.entities.config_entity import OPS_FILE_PATH, OPS_TRACE_FAILED_KEY
from core.ops.entities.trace_entity import trace_info_info_map
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models.model import Message
from models.workflow import WorkflowRun

# cdg:处理追踪任务，此处@shared_task装饰器用于将函数注册为Celery任务，并指定任务队列为ops_trace。
@shared_task(queue="ops_trace")
def process_trace_tasks(file_info):
    """
    Async process trace tasks
    :param tasks_data: List of dictionaries containing task data

    Usage: process_trace_tasks.delay(tasks_data)
    """
    from core.ops.ops_trace_manager import OpsTraceManager

    app_id = file_info.get("app_id")
    file_id = file_info.get("file_id")
    file_path = f"{OPS_FILE_PATH}{app_id}/{file_id}.json"
    file_data = json.loads(storage.load(file_path))
    trace_info = file_data.get("trace_info")
    trace_info_type = file_data.get("trace_info_type")

    # cdg: 获取操作跟踪实例。
    trace_instance = OpsTraceManager.get_ops_trace_instance(app_id)

    # cdg: 如果消息数据存在，则从字典转换为消息对象。
    if trace_info.get("message_data"):
        trace_info["message_data"] = Message.from_dict(data=trace_info["message_data"])
    # cdg: 如果工作流数据存在，则从字典转换为工作流运行对象。
    if trace_info.get("workflow_data"):
        trace_info["workflow_data"] = WorkflowRun.from_dict(data=trace_info["workflow_data"])
    # cdg: 如果文档数据存在，则从字典转换为文档对象列表。
    if trace_info.get("documents"):
        trace_info["documents"] = [Document(**doc) for doc in trace_info["documents"]]

    # cdg: 尝试处理操作跟踪。
    try:
        # cdg: 如果操作跟踪实例存在，则处理操作跟踪。
        if trace_instance:
            with current_app.app_context():
                # cdg: 根据操作跟踪类型获取操作跟踪类型实例。
                trace_type = trace_info_info_map.get(trace_info_type)
                if trace_type:
                    # cdg: 根据操作跟踪类型实例化操作跟踪对象。
                    trace_info = trace_type(**trace_info)
                # cdg: 处理操作跟踪。
                trace_instance.trace(trace_info)
        logging.info(f"Processing trace tasks success, app_id: {app_id}")
    except Exception:
        failed_key = f"{OPS_TRACE_FAILED_KEY}_{app_id}"
        redis_client.incr(failed_key)
        logging.info(f"Processing trace tasks failed, app_id: {app_id}")
    finally:
        # cdg: 删除操作跟踪文件。
        storage.delete(file_path)
