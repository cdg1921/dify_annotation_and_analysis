from typing import Optional

from core.ops.ops_trace_manager import OpsTraceManager, provider_config_map
from extensions.ext_database import db
from models.model import App, TraceAppConfig

# cdg:运维服务，主要用于获取和更新追踪应用配置。该功能默认关闭，需要手动开启。
class OpsService:
    # cdg:获取追踪应用配置，具体实现思路：
    # 1. 根据应用ID和追踪提供者获取追踪应用配置
    # 2. 如果追踪应用配置不存在，则返回None
    # 3. 解密追踪应用配置
    # 4. 获取追踪应用配置的URL
    # 5. 返回追踪应用配置
    @classmethod
    def get_tracing_app_config(cls, app_id: str, tracing_provider: str):
        """
        Get tracing app config
        :param app_id: app id
        :param tracing_provider: tracing provider
        :return:
        """
        # cdg:根据应用ID和追踪提供者从数据库中获取追踪应用配置。TraceAppConfig表中的数据默认为空，所以需要判断是否存在。
        trace_config_data: Optional[TraceAppConfig] = (
            db.session.query(TraceAppConfig)
            .filter(TraceAppConfig.app_id == app_id, TraceAppConfig.tracing_provider == tracing_provider)
            .first()
        )

        if not trace_config_data:
            return None

        # cdg:解密追踪应用配置
        # decrypt_token and obfuscated_token
        tenant = db.session.query(App).filter(App.id == app_id).first()
        if not tenant:
            return None
        tenant_id = tenant.tenant_id
        # cdg:解密追踪应用配置
        decrypt_tracing_config = OpsTraceManager.decrypt_tracing_config(
            tenant_id, tracing_provider, trace_config_data.tracing_config
        )

        new_decrypt_tracing_config = OpsTraceManager.obfuscated_decrypt_token(tracing_provider, decrypt_tracing_config)

        if tracing_provider == "langfuse" and (
            "project_key" not in decrypt_tracing_config or not decrypt_tracing_config.get("project_key")
        ):
            try:
                project_key = OpsTraceManager.get_trace_config_project_key(decrypt_tracing_config, tracing_provider)
                new_decrypt_tracing_config.update(
                    {
                        "project_url": "{host}/project/{key}".format(
                            host=decrypt_tracing_config.get("host"), key=project_key
                        )
                    }
                )
            except Exception:
                new_decrypt_tracing_config.update(
                    {"project_url": "{host}/".format(host=decrypt_tracing_config.get("host"))}
                )

        if tracing_provider == "langsmith" and (
            "project_url" not in decrypt_tracing_config or not decrypt_tracing_config.get("project_url")
        ):
            try:
                project_url = OpsTraceManager.get_trace_config_project_url(decrypt_tracing_config, tracing_provider)
                new_decrypt_tracing_config.update({"project_url": project_url})
            except Exception:
                new_decrypt_tracing_config.update({"project_url": "https://smith.langchain.com/"})

        if tracing_provider == "opik" and (
            "project_url" not in decrypt_tracing_config or not decrypt_tracing_config.get("project_url")
        ):
            try:
                project_url = OpsTraceManager.get_trace_config_project_url(decrypt_tracing_config, tracing_provider)
                new_decrypt_tracing_config.update({"project_url": project_url})
            except Exception:
                new_decrypt_tracing_config.update({"project_url": "https://www.comet.com/opik/"})

        trace_config_data.tracing_config = new_decrypt_tracing_config
        return trace_config_data.to_dict()

    # cdg:创建追踪应用配置，具体实现思路：
    # 1. 检查追踪提供者是否有效
    # 2. 获取项目URL
    # 3. 检查追踪应用配置是否存在
    # 4. 获取租户ID
    # 5. 加密追踪应用配置
    # 6. 创建追踪应用配置
    @classmethod
    def create_tracing_app_config(cls, app_id: str, tracing_provider: str, tracing_config: dict):
        """
        Create tracing app config
        :param app_id: app id
        :param tracing_provider: tracing provider
        :param tracing_config: tracing config
        :return:
        """
        # cdg:检查追踪提供者是否有效
        if tracing_provider not in provider_config_map and tracing_provider:
            return {"error": f"Invalid tracing provider: {tracing_provider}"}

        # cdg:获取追踪提供者配置
        config_class, other_keys = (
            provider_config_map[tracing_provider]["config_class"],
            provider_config_map[tracing_provider]["other_keys"],
        )
        # FIXME: ignore type error
        default_config_instance = config_class(**tracing_config)  # type: ignore
        for key in other_keys:  # type: ignore
            if key in tracing_config and tracing_config[key] == "":
                tracing_config[key] = getattr(default_config_instance, key, None)

        # api check
        if not OpsTraceManager.check_trace_config_is_effective(tracing_config, tracing_provider):
            return {"error": "Invalid Credentials"}

        # get project url
        if tracing_provider == "langfuse":
            project_key = OpsTraceManager.get_trace_config_project_key(tracing_config, tracing_provider)
            project_url = "{host}/project/{key}".format(host=tracing_config.get("host"), key=project_key)
        elif tracing_provider in ("langsmith", "opik"):
            project_url = OpsTraceManager.get_trace_config_project_url(tracing_config, tracing_provider)
        else:
            project_url = None

        # check if trace config already exists
        trace_config_data: Optional[TraceAppConfig] = (
            db.session.query(TraceAppConfig)
            .filter(TraceAppConfig.app_id == app_id, TraceAppConfig.tracing_provider == tracing_provider)
            .first()
        )

        if trace_config_data:
            return None

        # get tenant id
        tenant = db.session.query(App).filter(App.id == app_id).first()
        if not tenant:
            return None
        tenant_id = tenant.tenant_id
        tracing_config = OpsTraceManager.encrypt_tracing_config(tenant_id, tracing_provider, tracing_config)
        if project_url:
            tracing_config["project_url"] = project_url
        trace_config_data = TraceAppConfig(
            app_id=app_id,
            tracing_provider=tracing_provider,
            tracing_config=tracing_config,
        )
        db.session.add(trace_config_data)
        db.session.commit()

        return {"result": "success"}

    # cdg:更新追踪应用配置，具体实现思路：
    # 1. 检查追踪提供者是否有效
    # 2. 检查追踪应用配置是否存在
    # 3. 获取租户ID
    # 4. 加密追踪应用配置
    # 5. 检查追踪应用配置是否有效
    # 6. 更新追踪应用配置
    @classmethod
    def update_tracing_app_config(cls, app_id: str, tracing_provider: str, tracing_config: dict):
        """
        Update tracing app config
        :param app_id: app id
        :param tracing_provider: tracing provider
        :param tracing_config: tracing config
        :return:
        """
        if tracing_provider not in provider_config_map:
            raise ValueError(f"Invalid tracing provider: {tracing_provider}")

        # check if trace config already exists
        current_trace_config = (
            db.session.query(TraceAppConfig)
            .filter(TraceAppConfig.app_id == app_id, TraceAppConfig.tracing_provider == tracing_provider)
            .first()
        )

        if not current_trace_config:
            return None

        # get tenant id
        tenant = db.session.query(App).filter(App.id == app_id).first()
        if not tenant:
            return None
        tenant_id = tenant.tenant_id
        tracing_config = OpsTraceManager.encrypt_tracing_config(
            tenant_id, tracing_provider, tracing_config, current_trace_config.tracing_config
        )

        # api check
        # decrypt_token
        decrypt_tracing_config = OpsTraceManager.decrypt_tracing_config(tenant_id, tracing_provider, tracing_config)
        if not OpsTraceManager.check_trace_config_is_effective(decrypt_tracing_config, tracing_provider):
            raise ValueError("Invalid Credentials")

        current_trace_config.tracing_config = tracing_config
        db.session.commit()

        return current_trace_config.to_dict()

    # cdg:删除追踪应用配置，具体实现思路：
    # 1. 检查追踪应用配置是否存在
    # 2. 删除追踪应用配置
    # 3. 返回删除结果
    @classmethod
    def delete_tracing_app_config(cls, app_id: str, tracing_provider: str):
        """
        Delete tracing app config
        :param app_id: app id
        :param tracing_provider: tracing provider
        :return:
        """
        trace_config = (
            db.session.query(TraceAppConfig)
            .filter(TraceAppConfig.app_id == app_id, TraceAppConfig.tracing_provider == tracing_provider)
            .first()
        )

        if not trace_config:
            return None

        db.session.delete(trace_config)
        db.session.commit()

        return True
