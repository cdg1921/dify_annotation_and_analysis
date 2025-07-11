# cdg: 模型供应商管理模块相关表结构定义
from enum import Enum

from sqlalchemy import func

from .engine import db
from .types import StringUUID

# cdg: 定义ProviderType枚举，表示供应商类型，包括custom和system。
class ProviderType(Enum):
    CUSTOM = "custom"
    SYSTEM = "system"

    @staticmethod
    def value_of(value):
        for member in ProviderType:
            if member.value == value:
                return member
        raise ValueError(f"No matching enum found for value '{value}'")

# cdg: 定义ProviderQuotaType枚举，表示供应商配额类型，包括paid（付费）、free（免费）和trial（试用）。
class ProviderQuotaType(Enum):
    PAID = "paid"
    """hosted paid quota"""

    FREE = "free"
    """third-party free quota"""

    TRIAL = "trial"#
    """hosted trial quota"""

    @staticmethod
    def value_of(value):
        for member in ProviderQuotaType:
            if member.value == value:
                return member
        raise ValueError(f"No matching enum found for value '{value}'")

# cdg: 定义Provider模型，表示供应商信息，包括供应商ID、租户ID、供应商名称、供应商类型、加密配置、是否有效、最后使用时间、配额类型、配额限制、配额使用量、创建时间、更新时间。
class Provider(db.Model):  # type: ignore[name-defined]
    """
    Provider model representing the API providers and their configurations.
    """

    __tablename__ = "providers"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="provider_pkey"),
        db.Index("provider_tenant_id_provider_idx", "tenant_id", "provider_name"),
        db.UniqueConstraint(
            "tenant_id", "provider_name", "provider_type", "quota_type", name="unique_provider_name_type_quota"
        ),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    provider_type = db.Column(db.String(40), nullable=False, server_default=db.text("'custom'::character varying"))
    encrypted_config = db.Column(db.Text, nullable=True)
    is_valid = db.Column(db.Boolean, nullable=False, server_default=db.text("false"))
    last_used = db.Column(db.DateTime, nullable=True)

    quota_type = db.Column(db.String(40), nullable=True, server_default=db.text("''::character varying"))
    quota_limit = db.Column(db.BigInteger, nullable=True)
    quota_used = db.Column(db.BigInteger, default=0)

    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

    # cdg: 定义__repr__方法，用于打印供应商信息。
    def __repr__(self):
        return (
            f"<Provider(id={self.id}, tenant_id={self.tenant_id}, provider_name='{self.provider_name}',"
            f" provider_type='{self.provider_type}')>"
        )

    # cdg: 定义token_is_set属性，用于判断是否设置了令牌。
    @property
    def token_is_set(self):
        """
        Returns True if the encrypted_config is not None, indicating that the token is set.
        """
        return self.encrypted_config is not None

    # cdg: 定义is_enabled属性，用于判断是否启用供应商。
    @property
    def is_enabled(self):
        """
        Returns True if the provider is enabled.
        """
        if self.provider_type == ProviderType.SYSTEM.value:
            return self.is_valid
        else:
            return self.is_valid and self.token_is_set

# cdg: 定义ProviderModel模型，表示模型供应商信息，包括模型ID、租户ID、供应商名称、模型名称、模型类型、加密配置、是否有效、创建时间、更新时间。
class ProviderModel(db.Model):  # type: ignore[name-defined]
    """
    Provider model representing the API provider_models and their configurations.
    """

    __tablename__ = "provider_models"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="provider_model_pkey"),
        db.Index("provider_model_tenant_id_provider_idx", "tenant_id", "provider_name"),
        db.UniqueConstraint(
            "tenant_id", "provider_name", "model_name", "model_type", name="unique_provider_model_name"
        ),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(40), nullable=False)
    encrypted_config = db.Column(db.Text, nullable=True)
    is_valid = db.Column(db.Boolean, nullable=False, server_default=db.text("false"))
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

# cdg: 定义TenantDefaultModel模型，表示租户默认模型信息，包括模型ID、租户ID、供应商名称、模型名称、模型类型、创建时间、更新时间。
class TenantDefaultModel(db.Model):  # type: ignore[name-defined]
    __tablename__ = "tenant_default_models"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="tenant_default_model_pkey"),
        db.Index("tenant_default_model_tenant_id_provider_type_idx", "tenant_id", "provider_name", "model_type"),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

# cdg: 定义TenantPreferredModelProvider模型，表示租户首选模型供应商信息，包括模型ID、租户ID、供应商名称、首选供应商类型、创建时间、更新时间。
class TenantPreferredModelProvider(db.Model):  # type: ignore[name-defined]
    __tablename__ = "tenant_preferred_model_providers"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="tenant_preferred_model_provider_pkey"),
        db.Index("tenant_preferred_model_provider_tenant_provider_idx", "tenant_id", "provider_name"),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    preferred_provider_type = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

# cdg: 定义ProviderOrder模型，表示供应商订单信息，包括订单ID、租户ID、供应商名称、账户ID、支付产品ID、支付ID、交易ID、数量、货币、总金额、支付状态、支付时间、支付失败时间、退款时间、创建时间、更新时间。
class ProviderOrder(db.Model):  # type: ignore[name-defined]
    __tablename__ = "provider_orders"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="provider_order_pkey"),
        db.Index("provider_order_tenant_provider_idx", "tenant_id", "provider_name"),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    account_id = db.Column(StringUUID, nullable=False)
    payment_product_id = db.Column(db.String(191), nullable=False)
    payment_id = db.Column(db.String(191))
    transaction_id = db.Column(db.String(191))
    quantity = db.Column(db.Integer, nullable=False, server_default=db.text("1"))
    currency = db.Column(db.String(40))
    total_amount = db.Column(db.Integer)
    payment_status = db.Column(db.String(40), nullable=False, server_default=db.text("'wait_pay'::character varying"))
    paid_at = db.Column(db.DateTime)
    pay_failed_at = db.Column(db.DateTime)
    refunded_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

# cdg: 定义ProviderModelSetting模型，表示模型供应商设置信息，包括设置ID、租户ID、供应商名称、模型名称、模型类型、是否启用、负载均衡是否启用、创建时间、更新时间。
class ProviderModelSetting(db.Model):  # type: ignore[name-defined]
    """
    Provider model settings for record the model enabled status and load balancing status.
    """

    __tablename__ = "provider_model_settings"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="provider_model_setting_pkey"),
        db.Index("provider_model_setting_tenant_provider_model_idx", "tenant_id", "provider_name", "model_type"),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(40), nullable=False)
    enabled = db.Column(db.Boolean, nullable=False, server_default=db.text("true"))
    load_balancing_enabled = db.Column(db.Boolean, nullable=False, server_default=db.text("false"))
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())

# cdg: 定义LoadBalancingModelConfig模型，表示负载均衡模型配置信息，包括配置ID、租户ID、供应商名称、模型名称、模型类型、名称、加密配置、是否启用、创建时间、更新时间。
class LoadBalancingModelConfig(db.Model):  # type: ignore[name-defined]
    """
    Configurations for load balancing models.
    """

    __tablename__ = "load_balancing_model_configs"
    __table_args__ = (
        db.PrimaryKeyConstraint("id", name="load_balancing_model_config_pkey"),
        db.Index("load_balancing_model_config_tenant_provider_model_idx", "tenant_id", "provider_name", "model_type"),
    )

    id = db.Column(StringUUID, server_default=db.text("uuid_generate_v4()"))
    tenant_id = db.Column(StringUUID, nullable=False)
    provider_name = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(40), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    encrypted_config = db.Column(db.Text, nullable=True)
    enabled = db.Column(db.Boolean, nullable=False, server_default=db.text("true"))
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
