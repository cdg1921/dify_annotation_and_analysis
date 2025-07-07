from events.app_event import app_was_created
from extensions.ext_database import db
from models.model import InstalledApp

# cdg: 使用@app_was_created.connect装饰器，将handle函数注册为应用创建事件的处理器。
@app_was_created.connect
def handle(sender, **kwargs): # cdg: 处理应用创建事件，sender是应用对象，kwargs是事件参数
    """Create an installed app when an app is created.""" # cdg: 创建安装的应用
    app = sender # cdg: 获取应用对象
    installed_app = InstalledApp( # cdg: 创建安装的应用对象
        tenant_id=app.tenant_id, # cdg: 租户ID
        app_id=app.id, # cdg: 应用ID
        app_owner_tenant_id=app.tenant_id, # cdg: 应用所有者租户ID
    )
    db.session.add(installed_app) # cdg: 添加安装的应用对象到数据库
    db.session.commit() # cdg: 提交安装的应用对象到数据库
