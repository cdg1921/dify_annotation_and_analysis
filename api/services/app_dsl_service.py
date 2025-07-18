import logging
import uuid
from enum import StrEnum
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

import yaml  # type: ignore
from packaging import version
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.helper import ssrf_proxy
from events.app_event import app_model_config_was_updated, app_was_created
from extensions.ext_redis import redis_client
from factories import variable_factory
from models import Account, App, AppMode
from models.model import AppModelConfig
from services.workflow_service import WorkflowService

logger = logging.getLogger(__name__)

IMPORT_INFO_REDIS_KEY_PREFIX = "app_import_info:"
IMPORT_INFO_REDIS_EXPIRY = 180  # 3 minutes
CURRENT_DSL_VERSION = "0.1.5"

# cdg: 导入模式，包括YAML_CONTENT、YAML_URL两种模式。
class ImportMode(StrEnum):
    YAML_CONTENT = "yaml-content"
    YAML_URL = "yaml-url"

# cdg: 导入状态，包括COMPLETED、COMPLETED_WITH_WARNINGS、PENDING、FAILED四种状态，即完成、完成并警告、等待、失败。
class ImportStatus(StrEnum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed-with-warnings"
    PENDING = "pending"
    FAILED = "failed"

# cdg: 导入，包括ID、状态、应用ID、当前DSL版本、导入DSL版本、错误信息。
class Import(BaseModel):
    id: str
    status: ImportStatus
    app_id: Optional[str] = None
    current_dsl_version: str = CURRENT_DSL_VERSION
    imported_dsl_version: str = ""
    error: str = ""

# cdg: 检查版本兼容性，具体实现为：解析当前DSL版本和导入DSL版本；如果导入DSL版本解析失败，则返回失败；如果当前DSL版本和导入DSL版本的主版本和次版本不一致，则返回等待；如果当前DSL版本和导入DSL版本的主版本和次版本一致，则返回完成并警告；如果当前DSL版本和导入DSL版本的主版本和次版本一致，则返回完成。
def _check_version_compatibility(imported_version: str) -> ImportStatus:
    """Determine import status based on version comparison"""
    # cdg:基于版本号判断导入状态

    # cdg: 解析当前DSL版本和导入DSL版本
    try:
        current_ver = version.parse(CURRENT_DSL_VERSION)
        imported_ver = version.parse(imported_version)
    except version.InvalidVersion:
        return ImportStatus.FAILED

    # cdg: 比较主版本和次版本
    # Compare major version and minor version
    if current_ver.major != imported_ver.major or current_ver.minor != imported_ver.minor:
        return ImportStatus.PENDING

    # cdg: 比较微版本
    if current_ver.micro != imported_ver.micro:
        return ImportStatus.COMPLETED_WITH_WARNINGS

    return ImportStatus.COMPLETED

# cdg: 导入数据类，包括导入模式、YAML内容、名称、描述、图标类型、图标、图标背景、应用ID。
class PendingData(BaseModel):
    import_mode: str
    yaml_content: str
    name: str | None
    description: str | None
    icon_type: str | None
    icon: str | None
    icon_background: str | None
    app_id: str | None

# cdg: 应用DSL服务，包括导入应用、确认导入、创建或更新应用。
class AppDslService:
    def __init__(self, session: Session):
        self._session = session

    # cdg: 导入应用，具体实现为：验证导入模式；获取YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理YAML内容；处理Y
    def import_app(
        self,
        *,
        account: Account,
        import_mode: str,
        yaml_content: Optional[str] = None,
        yaml_url: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        icon_type: Optional[str] = None,
        icon: Optional[str] = None,
        icon_background: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Import:
        """Import an app from YAML content or URL."""
        import_id = str(uuid.uuid4())

        # Validate import mode
        try:
            mode = ImportMode(import_mode)
        except ValueError:
            raise ValueError(f"Invalid import_mode: {import_mode}")

        # cdg: 获取YAML内容
        # Get YAML content
        content: str = ""
        if mode == ImportMode.YAML_URL:
            if not yaml_url:
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error="yaml_url is required when import_mode is yaml-url",
                )
            try:
                max_size = 10 * 1024 * 1024  # 10MB
                parsed_url = urlparse(yaml_url)
                if (
                    parsed_url.scheme == "https"
                    and parsed_url.netloc == "github.com"
                    and parsed_url.path.endswith((".yml", ".yaml"))
                ):
                    yaml_url = yaml_url.replace("https://github.com", "https://raw.githubusercontent.com")
                    yaml_url = yaml_url.replace("/blob/", "/")
                response = ssrf_proxy.get(yaml_url.strip(), follow_redirects=True, timeout=(10, 10))
                response.raise_for_status()
                content = response.content.decode()

                if len(content) > max_size:
                    return Import(
                        id=import_id,
                        status=ImportStatus.FAILED,
                        error="File size exceeds the limit of 10MB",
                    )

                if not content:
                    return Import(
                        id=import_id,
                        status=ImportStatus.FAILED,
                        error="Empty content from url",
                    )
            except Exception as e:
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error=f"Error fetching YAML from URL: {str(e)}",
                )
        elif mode == ImportMode.YAML_CONTENT:
            if not yaml_content:
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error="yaml_content is required when import_mode is yaml-content",
                )
            content = yaml_content

        # cdg: 处理YAML内容
        # Process YAML content
        try:
            # Parse YAML to validate format
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error="Invalid YAML format: content must be a mapping",
                )

            # Validate and fix DSL version
            if not data.get("version"):
                data["version"] = "0.1.0"
            if not data.get("kind") or data.get("kind") != "app":
                data["kind"] = "app"

            imported_version = data.get("version", "0.1.0")
            # check if imported_version is a float-like string
            if not isinstance(imported_version, str):
                raise ValueError(f"Invalid version type, expected str, got {type(imported_version)}")
            status = _check_version_compatibility(imported_version)

            # Extract app data
            app_data = data.get("app")
            if not app_data:
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error="Missing app data in YAML content",
                )

            # If app_id is provided, check if it exists
            app = None
            if app_id:
                stmt = select(App).where(App.id == app_id, App.tenant_id == account.current_tenant_id)
                app = self._session.scalar(stmt)

                if not app:
                    return Import(
                        id=import_id,
                        status=ImportStatus.FAILED,
                        error="App not found",
                    )

                if app.mode not in [AppMode.WORKFLOW.value, AppMode.ADVANCED_CHAT.value]:
                    return Import(
                        id=import_id,
                        status=ImportStatus.FAILED,
                        error="Only workflow or advanced chat apps can be overwritten",
                    )

            # cdg: 如果主版本不匹配，则将导入信息存储在Redis中
            # If major version mismatch, store import info in Redis
            if status == ImportStatus.PENDING:
                panding_data = PendingData(
                    import_mode=import_mode,
                    yaml_content=content,
                    name=name,
                    description=description,
                    icon_type=icon_type,
                    icon=icon,
                    icon_background=icon_background,
                    app_id=app_id,
                )
                redis_client.setex(
                    f"{IMPORT_INFO_REDIS_KEY_PREFIX}{import_id}",
                    IMPORT_INFO_REDIS_EXPIRY,
                    panding_data.model_dump_json(),
                )

                return Import(
                    id=import_id,
                    status=status,
                    app_id=app_id,
                    imported_dsl_version=imported_version,
                )

            # cdg: 创建或更新应用
            # Create or update app
            app = self._create_or_update_app(
                app=app,
                data=data,
                account=account,
                name=name,
                description=description,
                icon_type=icon_type,
                icon=icon,
                icon_background=icon_background,
            )


            return Import(
                id=import_id,
                status=status,
                app_id=app.id,
                imported_dsl_version=imported_version,
            )

        except yaml.YAMLError as e:
            return Import(
                id=import_id,
                status=ImportStatus.FAILED,
                error=f"Invalid YAML format: {str(e)}",
            )

        except Exception as e:
            logger.exception("Failed to import app")
            return Import(
                id=import_id,
                status=ImportStatus.FAILED,
                error=str(e),
            )

    # cdg: 确认导入，具体实现为：获取导入信息；如果导入信息不存在，则返回失败；如果导入信息类型不正确，则返回失败；如果导入信息类型正确，则获取导入信息；如果导入信息应用ID存在，则获取应用；创建或更新应用；删除导入信息；返回导入信息。
    def confirm_import(self, *, import_id: str, account: Account) -> Import:
        """
        Confirm an import that requires confirmation
        """
        redis_key = f"{IMPORT_INFO_REDIS_KEY_PREFIX}{import_id}"
        pending_data = redis_client.get(redis_key)

        if not pending_data:
            return Import(
                id=import_id,
                status=ImportStatus.FAILED,
                error="Import information expired or does not exist",
            )

        try:
            if not isinstance(pending_data, str | bytes):
                return Import(
                    id=import_id,
                    status=ImportStatus.FAILED,
                    error="Invalid import information",
                )
            pending_data = PendingData.model_validate_json(pending_data)
            data = yaml.safe_load(pending_data.yaml_content)

            app = None
            if pending_data.app_id:
                stmt = select(App).where(App.id == pending_data.app_id, App.tenant_id == account.current_tenant_id)
                app = self._session.scalar(stmt)

            # Create or update app
            app = self._create_or_update_app(
                app=app,
                data=data,
                account=account,
                name=pending_data.name,
                description=pending_data.description,
                icon_type=pending_data.icon_type,
                icon=pending_data.icon,
                icon_background=pending_data.icon_background,
            )

            # Delete import info from Redis
            redis_client.delete(redis_key)

            return Import(
                id=import_id,
                status=ImportStatus.COMPLETED,
                app_id=app.id,
                current_dsl_version=CURRENT_DSL_VERSION,
                imported_dsl_version=data.get("version", "0.1.0"),
            )

        except Exception as e:
            logger.exception("Error confirming import")
            return Import(
                id=import_id,
                status=ImportStatus.FAILED,
                error=str(e),
            )

    # cdg: 创建或更新应用，具体实现为：获取应用数据；获取应用模式；如果应用模式不存在，则抛出异常；设置图标类型；设置图标；如果应用存在，则更新应用；否则创建应用；初始化应用；如果应用模式为工作流，则初始化工作流；如果应用模式为聊天，则初始化聊天；如果应用模式为完成，则初始化完成；返回应用。
    def _create_or_update_app(
        self,
        *,
        app: Optional[App],
        data: dict,
        account: Account,
        name: Optional[str] = None,
        description: Optional[str] = None,
        icon_type: Optional[str] = None,
        icon: Optional[str] = None,
        icon_background: Optional[str] = None,
    ) -> App:
        """Create a new app or update an existing one."""
        app_data = data.get("app", {})
        app_mode = app_data.get("mode")
        if not app_mode:
            raise ValueError("loss app mode")
        app_mode = AppMode(app_mode)

        # Set icon type
        icon_type_value = icon_type or app_data.get("icon_type")
        if icon_type_value in ["emoji", "link"]:
            icon_type = icon_type_value
        else:
            icon_type = "emoji"
        icon = icon or str(app_data.get("icon", ""))

        if app:
            # Update existing app
            app.name = name or app_data.get("name", app.name)
            app.description = description or app_data.get("description", app.description)
            app.icon_type = icon_type
            app.icon = icon
            app.icon_background = icon_background or app_data.get("icon_background", app.icon_background)
            app.updated_by = account.id
        else:
            if account.current_tenant_id is None:
                raise ValueError("Current tenant is not set")

            # Create new app
            app = App()
            app.id = str(uuid4())
            app.tenant_id = account.current_tenant_id
            app.mode = app_mode.value
            app.name = name or app_data.get("name", "")
            app.description = description or app_data.get("description", "")
            app.icon_type = icon_type
            app.icon = icon
            app.icon_background = icon_background or app_data.get("icon_background", "#FFFFFF")
            app.enable_site = True
            app.enable_api = True
            app.use_icon_as_answer_icon = app_data.get("use_icon_as_answer_icon", False)
            app.created_by = account.id
            app.updated_by = account.id

            self._session.add(app)
            self._session.commit()
            app_was_created.send(app, account=account)

        # Initialize app based on mode
        # cdg: 如果应用模式为工作流，则初始化工作流；如果应用模式为聊天，则初始化聊天；如果应用模式为完成，则初始化完成。
        if app_mode in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
            workflow_data = data.get("workflow")
            if not workflow_data or not isinstance(workflow_data, dict):
                raise ValueError("Missing workflow data for workflow/advanced chat app")

            # cdg: 初始化环境变量，从工作流数据中获取环境变量列表；如果环境变量列表不存在，则抛出异常；如果环境变量列表类型不正确，则抛出异常；初始化环境变量；初始化会话变量；初始化工作流服务；获取当前草稿工作流；如果当前草稿工作流存在，则获取当前草稿工作流的唯一哈希；否则设置为None；同步工作流；如果应用模式为聊天，则初始化聊天；如果应用模式为完成，则初始化完成。
            environment_variables_list = workflow_data.get("environment_variables", [])
            environment_variables = [
                variable_factory.build_environment_variable_from_mapping(obj) for obj in environment_variables_list
            ]

            # cdg: 初始化会话变量，从工作流数据中获取会话变量列表；如果会话变量列表不存在，则抛出异常；如果会话变量列表类型不正确，则抛出异常；初始化会话变量。
            conversation_variables_list = workflow_data.get("conversation_variables", [])
            conversation_variables = [
                variable_factory.build_conversation_variable_from_mapping(obj) for obj in conversation_variables_list
            ]
            # cdg:环境变量与会话变量的区别在于：环境变量是全局变量，会话变量是会话变量。

            # cdg: 初始化工作流服务；获取当前草稿工作流；如果当前草稿工作流存在，则获取当前草稿工作流的唯一哈希；否则设置为None；同步工作流；如果应用模式为聊天，则初始化聊天；如果应用模式为完成，则初始化完成。
            workflow_service = WorkflowService()
            current_draft_workflow = workflow_service.get_draft_workflow(app_model=app)
            if current_draft_workflow:
                unique_hash = current_draft_workflow.unique_hash
            else:
                unique_hash = None

            # cdg: 同步工作流；如果应用模式为聊天，则初始化聊天；如果应用模式为完成，则初始化完成。
            workflow_service.sync_draft_workflow(
                app_model=app,
                graph=workflow_data.get("graph", {}),
                features=workflow_data.get("features", {}),
                unique_hash=unique_hash,
                account=account,
                environment_variables=environment_variables,
                conversation_variables=conversation_variables,
            )
        elif app_mode in {AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.COMPLETION}:
            # Initialize model config
            model_config = data.get("model_config")
            if not model_config or not isinstance(model_config, dict):
                raise ValueError("Missing model_config for chat/agent-chat/completion app")
            # Initialize or update model config
            if not app.app_model_config:
                app_model_config = AppModelConfig().from_model_config_dict(model_config)
                app_model_config.id = str(uuid4())
                app_model_config.app_id = app.id
                app_model_config.created_by = account.id
                app_model_config.updated_by = account.id

                app.app_model_config_id = app_model_config.id

                self._session.add(app_model_config)
                app_model_config_was_updated.send(app, app_model_config=app_model_config)
        else:
            raise ValueError("Invalid app mode")
        return app

    # cdg: 导出DSL，具体实现为：获取应用模式；导出应用数据；如果应用模式为工作流，则导出工作流；如果应用模式为聊天，则导出聊天；如果应用模式为完成，则导出完成；返回DSL。
    @classmethod
    def export_dsl(cls, app_model: App, include_secret: bool = False) -> str:
        """
        Export app
        :param app_model: App instance
        :return:
        """
        app_mode = AppMode.value_of(app_model.mode)

        export_data = {
            "version": CURRENT_DSL_VERSION,
            "kind": "app",
            "app": {
                "name": app_model.name,
                "mode": app_model.mode,
                "icon": "🤖" if app_model.icon_type == "image" else app_model.icon,
                "icon_background": "#FFEAD5" if app_model.icon_type == "image" else app_model.icon_background,
                "description": app_model.description,
                "use_icon_as_answer_icon": app_model.use_icon_as_answer_icon,
            },
        }

        if app_mode in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
            cls._append_workflow_export_data(
                export_data=export_data, app_model=app_model, include_secret=include_secret
            )
        else:
            cls._append_model_config_export_data(export_data, app_model)

        return yaml.dump(export_data, allow_unicode=True)  # type: ignore

    # cdg: 导出工作流数据，具体实现为：获取工作流服务；获取当前草稿工作流；如果当前草稿工作流不存在，则抛出异常；导出工作流数据；返回工作流数据。

    @classmethod
    def _append_workflow_export_data(cls, *, export_data: dict, app_model: App, include_secret: bool) -> None:
        """
        Append workflow export data
        :param export_data: export data
        :param app_model: App instance
        """
        workflow_service = WorkflowService()
        workflow = workflow_service.get_draft_workflow(app_model)
        if not workflow:
            raise ValueError("Missing draft workflow configuration, please check.")

        export_data["workflow"] = workflow.to_dict(include_secret=include_secret)

    # cdg: 导出模型配置数据，具体实现为：获取应用模型配置；如果应用模型配置不存在，则抛出异常；导出模型配置数据；返回模型配置数据。
    @classmethod
    def _append_model_config_export_data(cls, export_data: dict, app_model: App) -> None:
        """
        Append model config export data
        :param export_data: export data
        :param app_model: App instance
        """
        app_model_config = app_model.app_model_config
        if not app_model_config:
            raise ValueError("Missing app configuration, please check.")

        export_data["model_config"] = app_model_config.to_dict()
