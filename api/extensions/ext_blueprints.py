from configs import dify_config
from dify_app import DifyApp

# cdg: 初始化蓝图
def init_app(app: DifyApp):
    # register blueprint routers
    # cdg: 导入CORS
    from flask_cors import CORS  # type: ignore
    # cdg: CORS是一种浏览器的安全机制，用于允许或限制不同源（域名、协议、端口）之间的资源请求。
    # 默认情况下，浏览器出于安全考虑，禁止网页从一个域去请求另一个域的资源（即“同源策略”）。
    # CORS 允许服务器通过设置 HTTP 响应头，明确告诉浏览器哪些跨域请求是被允许的。

    # cdg: 导入蓝图Blueprint（蓝图）是 Flask 提供的一种组件化管理路由和功能的机制。
    # 它允许你将应用拆分为多个模块，每个模块可以有自己的路由、视图、静态文件等，最后统一注册到主应用上。
    
    # cdg:CORS 解决前端与后端跨域访问的问题，通过设置响应头允许特定的跨域请求。
    # 蓝图是 Flask 的模块化机制，用于组织和管理大型项目的路由和功能。
    # 在实际项目中，常常会为不同的蓝图配置不同的 CORS 策略，以满足不同模块的安全和访问需求。
    from controllers.console import bp as console_app_bp
    from controllers.files import bp as files_bp
    # cdg: 导入内部API蓝图
    from controllers.inner_api import bp as inner_api_bp
    # cdg: 导入服务API蓝图
    from controllers.service_api import bp as service_api_bp
    # cdg: 导入Web API蓝图
    from controllers.web import bp as web_bp

    # cdg: 注册服务API蓝图
    CORS(
        service_api_bp, #cdg: 服务API蓝图
        allow_headers=["Content-Type", "Authorization", "X-App-Code"], #cdg: 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],  #cdg: 允许的请求方法
    )
    app.register_blueprint(service_api_bp) #cdg: 注册服务API蓝图

    # cdg: 注册Web API蓝图
    CORS(
        web_bp, #cdg: Web API蓝图
        resources={r"/*": {"origins": dify_config.WEB_API_CORS_ALLOW_ORIGINS}}, #cdg: 允许的请求源
        supports_credentials=True, #cdg: 支持凭证
        allow_headers=["Content-Type", "Authorization", "X-App-Code"], #cdg: 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"], #cdg: 允许的请求方法
        expose_headers=["X-Version", "X-Env"], #cdg: 暴露的响应头
    )

    app.register_blueprint(web_bp) #cdg: 注册Web API蓝图

    # cdg: 注册控制台API蓝图
    CORS(
        console_app_bp, #cdg: 控制台API蓝图
        resources={r"/*": {"origins": dify_config.CONSOLE_CORS_ALLOW_ORIGINS}}, #cdg: 允许的请求源
        supports_credentials=True, #cdg: 支持凭证
        allow_headers=["Content-Type", "Authorization"], #cdg: 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"], #cdg: 允许的请求方法
        expose_headers=["X-Version", "X-Env"], #cdg: 暴露的响应头
    )   

    app.register_blueprint(console_app_bp) #cdg: 注册控制台API蓝图

    # cdg: 注册文件API蓝图
    CORS(files_bp, allow_headers=["Content-Type"], methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"]) #cdg: 文件API蓝图
    app.register_blueprint(files_bp)

    # cdg: 注册内部API蓝图
    app.register_blueprint(inner_api_bp)
