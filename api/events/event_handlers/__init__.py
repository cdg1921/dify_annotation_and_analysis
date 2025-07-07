# cdg: 事件处理包，用于处理事件，从而实现事件的响应。
from .clean_when_dataset_deleted import handle # cdg: 当数据集删除时，清理数据集
from .clean_when_document_deleted import handle # cdg: 当文档删除时，清理文档
from .create_document_index import handle # cdg: 当文档创建时，创建文档索引
from .create_installed_app_when_app_created import handle # cdg: 当应用创建时，创建安装的应用
from .create_site_record_when_app_created import handle # cdg: 当应用创建时，创建站点记录
from .deduct_quota_when_message_created import handle # cdg: 当消息创建时，扣除配额
from .delete_tool_parameters_cache_when_sync_draft_workflow import handle # cdg: 当同步草稿工作流时，删除工具参数缓存
from .update_app_dataset_join_when_app_model_config_updated import handle # cdg: 当应用模型配置更新时，更新应用数据集关联
from .update_app_dataset_join_when_app_published_workflow_updated import handle # cdg: 当应用发布工作流更新时，更新应用数据集关联
from .update_provider_last_used_at_when_message_created import handle # cdg: 当消息创建时，更新提供者最后使用时间
