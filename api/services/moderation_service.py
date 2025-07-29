from typing import Optional

from core.moderation.factory import ModerationFactory, ModerationOutputsResult
from extensions.ext_database import db
from models.model import App, AppModelConfig

# cdg:内容审核服务
class ModerationService:
    # cdg:输出内容审核服务,具体实现思路：
    # 1. 根据应用ID和应用模型获取应用模型配置
    # 2. 根据应用模型配置获取内容审核配置
    # 3. 根据内容审核配置创建内容审核实例
    # 4. 执行内容审核，并返回内容审核结果
    def moderation_for_outputs(self, app_id: str, app_model: App, text: str) -> ModerationOutputsResult:
        # cdg:根据应用ID和应用模型获取应用模型配置
        app_model_config: Optional[AppModelConfig] = None

        # cdg:根据应用模型配置获取内容审核配置
        app_model_config = (
            db.session.query(AppModelConfig).filter(AppModelConfig.id == app_model.app_model_config_id).first()
        )

        # cdg:如果应用模型配置不存在，则抛出异常
        if not app_model_config:
            raise ValueError("app model config not found")

        # cdg:获取内容审核配置
        name = app_model_config.sensitive_word_avoidance_dict["type"]
        config = app_model_config.sensitive_word_avoidance_dict["config"]

        # cdg:创建内容审核实例
        moderation = ModerationFactory(name, app_id, app_model.tenant_id, config)

        # cdg:执行内容审核，并返回内容审核结果
        return moderation.moderation_for_outputs(text)
