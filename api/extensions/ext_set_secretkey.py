from configs import dify_config
from dify_app import DifyApp


def init_app(app: DifyApp): 
    # cdg: 初始化Flask应用的密钥，用于加密Flask应用的会话，从而保证会话的安全性。
    app.secret_key = dify_config.SECRET_KEY
