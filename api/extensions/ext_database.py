from dify_app import DifyApp
from models import db

# cdg: 初始化数据库
def init_app(app: DifyApp):
    db.init_app(app)
