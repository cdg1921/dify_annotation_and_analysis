from events.app_event import app_was_created
from extensions.ext_database import db
from models.model import Site

# cdg: 使用@app_was_created.connect装饰器，将handle函数注册为应用创建事件的处理器。
@app_was_created.connect
def handle(sender, **kwargs): # cdg: 处理应用创建事件，sender是应用对象，kwargs是事件参数   
    """Create site record when an app is created.""" # cdg: 创建站点记录
    app = sender # cdg: 获取应用对象
    account = kwargs.get("account") # cdg: 获取账户对象
    if account is not None: # cdg: 如果账户对象不为空，则创建站点记录
        site = Site( # cdg: 创建站点记录对象
            app_id=app.id, # cdg: 应用ID
            title=app.name, # cdg: 应用名称
            icon_type=app.icon_type, # cdg: 应用图标类型
            icon=app.icon, # cdg: 应用图标
            icon_background=app.icon_background, # cdg: 应用图标背景
            default_language=account.interface_language, # cdg: 默认语言
            customize_token_strategy="not_allow", # cdg: 自定义令牌策略
            code=Site.generate_code(16), # cdg: 生成站点代码
            created_by=app.created_by, # cdg: 创建者
            updated_by=app.updated_by, # cdg: 更新者
        )
        db.session.add(site) # cdg: 添加站点记录对象到数据库
        db.session.commit() # cdg: 提交站点记录对象到数据库

        db.session.add(site)
        db.session.commit()
