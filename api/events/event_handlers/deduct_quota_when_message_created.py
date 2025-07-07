from core.app.entities.app_invoke_entities import AgentChatAppGenerateEntity, ChatAppGenerateEntity
from core.entities.provider_entities import QuotaUnit
from events.message_event import message_was_created
from extensions.ext_database import db
from models.provider import Provider, ProviderType

# cdg: 使用@message_was_created.connect装饰器，将handle函数注册为消息创建事件的处理器,每一次消息创建时，都会调用该函数，扣除token额度
@message_was_created.connect
def handle(sender, **kwargs): # cdg: 处理消息创建事件，sender是消息对象，kwargs是事件参数
    message = sender # cdg: 获取消息对象
    application_generate_entity = kwargs.get("application_generate_entity") # cdg: 获取应用生成实体

    if not isinstance(application_generate_entity, ChatAppGenerateEntity | AgentChatAppGenerateEntity): # cdg: 如果应用生成实体不是ChatAppGenerateEntity或AgentChatAppGenerateEntity，则返回
        return

    model_config = application_generate_entity.model_conf # cdg: 获取模型配置
    provider_model_bundle = model_config.provider_model_bundle # cdg: 获取模型包
    provider_configuration = provider_model_bundle.configuration # cdg: 获取提供者配置

    if provider_configuration.using_provider_type != ProviderType.SYSTEM: # cdg: 如果提供者类型不是SYSTEM，则返回
        return

    system_configuration = provider_configuration.system_configuration # cdg: 获取系统配置

    quota_unit = None # cdg: 初始化额度单位，按次数、token、积分等方式  
    for quota_configuration in system_configuration.quota_configurations: # cdg: 遍历系统配置的配额配置
        if quota_configuration.quota_type == system_configuration.current_quota_type: # cdg: 如果配额类型与系统配置的配额类型相同，则设置配额单位
            quota_unit = quota_configuration.quota_unit # cdg: 设置配额单位

            if quota_configuration.quota_limit == -1: # cdg: 如果配额限制为-1，则返回,表示不限制
                return

            break

    used_quota = None # cdg: 初始化已使用额度
    if quota_unit: # cdg: 如果配额单位不为空，则计算已使用额度
        if quota_unit == QuotaUnit.TOKENS: # cdg: 如果配额单位为TOKEN，则计算已使用额度
            used_quota = message.message_tokens + message.answer_tokens # cdg: 计算已使用额度
        elif quota_unit == QuotaUnit.CREDITS: # cdg: 如果配额单位为积分，则计算已使用额度
            used_quota = 1 # cdg: 设置已使用额度

            if "gpt-4" in model_config.model: # cdg: 如果模型为gpt-4，则设置已使用额度
                used_quota = 20 # cdg: 设置已使用额度
        else:
            used_quota = 1 # cdg: 设置已使用额度

    if used_quota is not None and system_configuration.current_quota_type is not None:
        db.session.query(Provider).filter(
            Provider.tenant_id == application_generate_entity.app_config.tenant_id,
            Provider.provider_name == model_config.provider,
            Provider.provider_type == ProviderType.SYSTEM.value,
            Provider.quota_type == system_configuration.current_quota_type.value,
            Provider.quota_limit > Provider.quota_used,
        ).update({"quota_used": Provider.quota_used + used_quota})
        db.session.commit()
