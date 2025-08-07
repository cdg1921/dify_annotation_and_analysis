from flask_login import current_user  # type: ignore

from configs import dify_config
from extensions.ext_database import db
from models.account import Tenant, TenantAccountJoin, TenantAccountJoinRole
from services.account_service import TenantService
from services.feature_service import FeatureService

# cdg: 工作空间服务，用于管理工作空间信息。
class WorkspaceService:
    @classmethod
    def get_tenant_info(cls, tenant: Tenant):
        # cdg: 获取当前用户的工作空间信息，一个工作空间对应一个租户。
        if not tenant:
            return None
        tenant_info = {
            "id": tenant.id,
            "name": tenant.name,
            "plan": tenant.plan,
            "status": tenant.status,
            "created_at": tenant.created_at,
            "in_trail": True,
            "trial_end_reason": None,
            "role": "normal",
        }

        # cdg: 获取当前用户在当前工作空间中的角色。
        # Get role of user
        tenant_account_join = (
            db.session.query(TenantAccountJoin)
            .filter(TenantAccountJoin.tenant_id == tenant.id, TenantAccountJoin.account_id == current_user.id)
            .first()
        )
        assert tenant_account_join is not None, "TenantAccountJoin not found"
        tenant_info["role"] = tenant_account_join.role

        # cdg: 获取当前工作空间是否可以替换logo。非授权用户不允许替换logo。
        can_replace_logo = FeatureService.get_features(tenant_info["id"]).can_replace_logo

        # cdg: 如果当前工作空间可以替换logo，则获取当前工作空间的logo。
        if can_replace_logo and TenantService.has_roles(
            tenant, [TenantAccountJoinRole.OWNER, TenantAccountJoinRole.ADMIN]
        ):
            # cdg: 获取当前工作空间的logo。
            base_url = dify_config.FILES_URL
            replace_webapp_logo = (
                f"{base_url}/files/workspaces/{tenant.id}/webapp-logo"
                if tenant.custom_config_dict.get("replace_webapp_logo")
                else None
            )
            # cdg: 获取当前工作空间是否可以删除logo。
            remove_webapp_brand = tenant.custom_config_dict.get("remove_webapp_brand", False)

            # cdg: 获取当前工作空间的logo。
            tenant_info["custom_config"] = {
                "remove_webapp_brand": remove_webapp_brand,
                "replace_webapp_logo": replace_webapp_logo,
            }

        return tenant_info
