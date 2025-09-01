import logging
import time
from collections.abc import Callable

import click
from celery import shared_task  # type: ignore
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError

from extensions.ext_database import db
from models.dataset import AppDatasetJoin
from models.model import (
    ApiToken,
    AppAnnotationHitHistory,
    AppAnnotationSetting,
    AppModelConfig,
    Conversation,
    EndUser,
    InstalledApp,
    Message,
    MessageAgentThought,
    MessageAnnotation,
    MessageChain,
    MessageFeedback,
    MessageFile,
    RecommendedApp,
    Site,
    TagBinding,
    TraceAppConfig,
)
from models.tools import WorkflowToolProvider
from models.web import PinnedConversation, SavedMessage
from models.workflow import ConversationVariable, Workflow, WorkflowAppLog, WorkflowNodeExecution, WorkflowRun

# cdg: 用于异步删除应用和相关数据. celery -A celery_app.celery_app.celery worker -Q app_deletion -n worker1@%h
@shared_task(queue="app_deletion", bind=True, max_retries=3)
def remove_app_and_related_data_task(self, tenant_id: str, app_id: str):
    logging.info(click.style(f"Start deleting app and related data: {tenant_id}:{app_id}", fg="green"))
    start_at = time.perf_counter()
    try:
        # Delete related data
        # cdg: 删除应用模型配置 1000条记录为一批次
        _delete_app_model_configs(tenant_id, app_id)
        # cdg: 删除应用站点
        _delete_app_site(tenant_id, app_id)
        # cdg: 删除应用API令牌
        _delete_app_api_tokens(tenant_id, app_id)
        # cdg: 删除已安装应用
        _delete_installed_apps(tenant_id, app_id)
        # cdg: 删除推荐应用
        _delete_recommended_apps(tenant_id, app_id)
        # cdg: 删除应用标注数据
        _delete_app_annotation_data(tenant_id, app_id)
        # cdg: 删除应用数据集关联
        _delete_app_dataset_joins(tenant_id, app_id)
        # cdg: 删除应用工作流
        _delete_app_workflows(tenant_id, app_id)
        # cdg: 删除应用工作流运行
        _delete_app_workflow_runs(tenant_id, app_id)
        # cdg: 删除应用工作流节点执行
        _delete_app_workflow_node_executions(tenant_id, app_id)
        # cdg: 删除应用工作流应用日志
        _delete_app_workflow_app_logs(tenant_id, app_id)
        # cdg: 删除应用对话
        _delete_app_conversations(tenant_id, app_id)
        # cdg: 删除应用消息
        _delete_app_messages(tenant_id, app_id)
        # cdg: 删除应用工作流工具提供者
        _delete_workflow_tool_providers(tenant_id, app_id)
        # cdg: 删除应用标签绑定
        _delete_app_tag_bindings(tenant_id, app_id)
        # cdg: 删除应用终端用户
        _delete_end_users(tenant_id, app_id)
        # cdg: 删除应用跟踪配置
        _delete_trace_app_configs(tenant_id, app_id)
        # cdg: 删除应用对话变量
        _delete_conversation_variables(app_id=app_id)

        end_at = time.perf_counter()
        # cdg: 记录删除应用和相关数据的时间,click.style用于美化日志输出,fg="green"表示绿色
        logging.info(click.style(f"App and related data deleted: {app_id} latency: {end_at - start_at}", fg="green"))
    except SQLAlchemyError as e:
        logging.exception(
            click.style(f"Database error occurred while deleting app {app_id} and related data", fg="red")
        )
        # cdg: 如果数据库错误,重试60秒
        raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds
    except Exception as e:
        logging.exception(click.style(f"Error occurred while deleting app {app_id} and related data", fg="red"))
        # cdg: 如果其他错误,重试60秒
        raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds


def _delete_app_model_configs(tenant_id: str, app_id: str):
    # cdg: 删除应用模型配置
    def del_model_config(model_config_id: str):
        db.session.query(AppModelConfig).filter(AppModelConfig.id == model_config_id).delete(synchronize_session=False)

    # cdg: 删除数据库中的app_model_configs表中的数据,1000条记录为一批次
    _delete_records(
        """select id from app_model_configs where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_model_config,
        "app model config",
    )


# cdg: 删除应用站点
def _delete_app_site(tenant_id: str, app_id: str):
    def del_site(site_id: str):
        db.session.query(Site).filter(Site.id == site_id).delete(synchronize_session=False)

    _delete_records("""select id from sites where app_id=:app_id limit 1000""", {"app_id": app_id}, del_site, "site")


def _delete_app_api_tokens(tenant_id: str, app_id: str):
    def del_api_token(api_token_id: str):
        db.session.query(ApiToken).filter(ApiToken.id == api_token_id).delete(synchronize_session=False)

    _delete_records(
        """select id from api_tokens where app_id=:app_id limit 1000""", {"app_id": app_id}, del_api_token, "api token"
    )


def _delete_installed_apps(tenant_id: str, app_id: str):
    def del_installed_app(installed_app_id: str):
        db.session.query(InstalledApp).filter(InstalledApp.id == installed_app_id).delete(synchronize_session=False)

    _delete_records(
        """select id from installed_apps where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_installed_app,
        "installed app",
    )


def _delete_recommended_apps(tenant_id: str, app_id: str):
    def del_recommended_app(recommended_app_id: str):
        db.session.query(RecommendedApp).filter(RecommendedApp.id == recommended_app_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from recommended_apps where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_recommended_app,
        "recommended app",
    )


def _delete_app_annotation_data(tenant_id: str, app_id: str):
    def del_annotation_hit_history(annotation_hit_history_id: str):
        db.session.query(AppAnnotationHitHistory).filter(
            AppAnnotationHitHistory.id == annotation_hit_history_id
        ).delete(synchronize_session=False)

    _delete_records(
        """select id from app_annotation_hit_histories where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_annotation_hit_history,
        "annotation hit history",
    )

    def del_annotation_setting(annotation_setting_id: str):
        db.session.query(AppAnnotationSetting).filter(AppAnnotationSetting.id == annotation_setting_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from app_annotation_settings where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_annotation_setting,
        "annotation setting",
    )


def _delete_app_dataset_joins(tenant_id: str, app_id: str):
    def del_dataset_join(dataset_join_id: str):
        db.session.query(AppDatasetJoin).filter(AppDatasetJoin.id == dataset_join_id).delete(synchronize_session=False)

    _delete_records(
        """select id from app_dataset_joins where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_dataset_join,
        "dataset join",
    )


def _delete_app_workflows(tenant_id: str, app_id: str):
    def del_workflow(workflow_id: str):
        db.session.query(Workflow).filter(Workflow.id == workflow_id).delete(synchronize_session=False)

    _delete_records(
        """select id from workflows where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_workflow,
        "workflow",
    )


def _delete_app_workflow_runs(tenant_id: str, app_id: str):
    def del_workflow_run(workflow_run_id: str):
        db.session.query(WorkflowRun).filter(WorkflowRun.id == workflow_run_id).delete(synchronize_session=False)

    _delete_records(
        """select id from workflow_runs where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_workflow_run,
        "workflow run",
    )


def _delete_app_workflow_node_executions(tenant_id: str, app_id: str):
    def del_workflow_node_execution(workflow_node_execution_id: str):
        db.session.query(WorkflowNodeExecution).filter(WorkflowNodeExecution.id == workflow_node_execution_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from workflow_node_executions where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_workflow_node_execution,
        "workflow node execution",
    )


def _delete_app_workflow_app_logs(tenant_id: str, app_id: str):
    def del_workflow_app_log(workflow_app_log_id: str):
        db.session.query(WorkflowAppLog).filter(WorkflowAppLog.id == workflow_app_log_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from workflow_app_logs where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_workflow_app_log,
        "workflow app log",
    )


def _delete_app_conversations(tenant_id: str, app_id: str):
    def del_conversation(conversation_id: str):
        db.session.query(PinnedConversation).filter(PinnedConversation.conversation_id == conversation_id).delete(
            synchronize_session=False
        )
        db.session.query(Conversation).filter(Conversation.id == conversation_id).delete(synchronize_session=False)

    _delete_records(
        """select id from conversations where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_conversation,
        "conversation",
    )


def _delete_conversation_variables(*, app_id: str):
    stmt = delete(ConversationVariable).where(ConversationVariable.app_id == app_id)
    with db.engine.connect() as conn:
        conn.execute(stmt)
        conn.commit()
        logging.info(click.style(f"Deleted conversation variables for app {app_id}", fg="green"))


def _delete_app_messages(tenant_id: str, app_id: str):
    def del_message(message_id: str):
        db.session.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).delete(
            synchronize_session=False
        )
        db.session.query(MessageAnnotation).filter(MessageAnnotation.message_id == message_id).delete(
            synchronize_session=False
        )
        db.session.query(MessageChain).filter(MessageChain.message_id == message_id).delete(synchronize_session=False)
        db.session.query(MessageAgentThought).filter(MessageAgentThought.message_id == message_id).delete(
            synchronize_session=False
        )
        db.session.query(MessageFile).filter(MessageFile.message_id == message_id).delete(synchronize_session=False)
        db.session.query(SavedMessage).filter(SavedMessage.message_id == message_id).delete(synchronize_session=False)
        db.session.query(Message).filter(Message.id == message_id).delete()

    _delete_records(
        """select id from messages where app_id=:app_id limit 1000""", {"app_id": app_id}, del_message, "message"
    )


def _delete_workflow_tool_providers(tenant_id: str, app_id: str):
    def del_tool_provider(tool_provider_id: str):
        db.session.query(WorkflowToolProvider).filter(WorkflowToolProvider.id == tool_provider_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from tool_workflow_providers where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_tool_provider,
        "tool workflow provider",
    )


def _delete_app_tag_bindings(tenant_id: str, app_id: str):
    def del_tag_binding(tag_binding_id: str):
        db.session.query(TagBinding).filter(TagBinding.id == tag_binding_id).delete(synchronize_session=False)

    _delete_records(
        """select id from tag_bindings where tenant_id=:tenant_id and target_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_tag_binding,
        "tag binding",
    )


def _delete_end_users(tenant_id: str, app_id: str):
    def del_end_user(end_user_id: str):
        db.session.query(EndUser).filter(EndUser.id == end_user_id).delete(synchronize_session=False)

    _delete_records(
        """select id from end_users where tenant_id=:tenant_id and app_id=:app_id limit 1000""",
        {"tenant_id": tenant_id, "app_id": app_id},
        del_end_user,
        "end user",
    )


def _delete_trace_app_configs(tenant_id: str, app_id: str):
    def del_trace_app_config(trace_app_config_id: str):
        db.session.query(TraceAppConfig).filter(TraceAppConfig.id == trace_app_config_id).delete(
            synchronize_session=False
        )

    _delete_records(
        """select id from trace_app_config where app_id=:app_id limit 1000""",
        {"app_id": app_id},
        del_trace_app_config,
        "trace app config",
    )

# cdg: 删除数据库中的数据,1000条记录为一批次
def _delete_records(query_sql: str, params: dict, delete_func: Callable, name: str) -> None:
    while True:
        # cdg: 使用db.engine.begin()作为上下文管理器,避免手动提交事务
        with db.engine.begin() as conn:
            # cdg: 执行查询语句,获取查询结果
            rs = conn.execute(db.text(query_sql), params)
            # cdg: 如果查询结果为0,则退出循环
            if rs.rowcount == 0:
                break

            # cdg: 遍历查询结果,删除每条记录
            for i in rs:
                record_id = str(i.id)
                try:
                    # cdg: 调用删除函数,删除每条记录
                    delete_func(record_id)
                    db.session.commit()
                    # cdg: 记录删除记录的时间
                    logging.info(click.style(f"Deleted {name} {record_id}", fg="green"))
                except Exception:
                    logging.exception(f"Error occurred while deleting {name} {record_id}")
                    continue
            rs.close()
