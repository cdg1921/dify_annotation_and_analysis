import logging
import time

from configs import dify_config
from dify_app import DifyApp


# ----------------------------
# Application Factory Function
# ----------------------------
def create_flask_app_with_configs() -> DifyApp:
    """
    create a raw flask app
    with configs loaded from .env file
    """
    # cdg:DifyApp是一个继承自flask.Flask的新类。
    # 通过继承，DifyApp类拥有了Flask类的所有功能和特性。
    # 此时，DifyApp可以用于创建Flask应用程序的实例，并可以在其中添加自定义的功能或方法。
    # 通常用的都是app = Flask(__name__)，这里只是特别指定为DIFY相关的APP而已，名称是可以自定义的
    dify_app = DifyApp(__name__)
    # cdg:dify_config是一个集成于pydantic.BaseModel子类的类，通过model_dump()将属性值转为字典格式。
    # 然后使用 from_mapping() 方法将字典加载配置到dify_app的配置对象中。
    # 通过这种方式，可以方便地将配置项（如数据库连接、调试模式等）应用到Flask应用中。
    dify_app.config.from_mapping(dify_config.model_dump())

    return dify_app


def create_app() -> DifyApp:
    # cdg:time.perf_counter()是一个高精度计时器，用于测量时间间隔。
    # 它返回自某个不确定的时间点（通常是程序启动时）以来的秒数，精度高于time.time()，适合用于性能测量和计时。
    # time.time() 和 time.perf_counter() 之间的主要区别如下：
    # （1）精度方面：
    # time.time()：返回自纪元（通常是1970年1月1日）以来的秒数，精度通常为秒级，具体取决于操作系统的实现。它适合用于获取当前时间，但不适合用于高精度计时。
    # time.perf_counter()：返回一个高精度的计时器值，表示自某个不确定的时间点（通常是程序启动时）以来的秒数。它的精度通常高于time.time()，适合用于测量时间间隔和性能分析。
    # （2）用途方面：
    # time.time()：常用于获取当前的时间戳，适合用于时间计算、日期处理等场景。
    # time.perf_counter()：主要用于性能测量和计时，适合用于需要高精度的时间间隔测量，例如代码执行时间的测量。
    # （3）稳定性方面：
    # time.time()：可能会受到系统时间调整（如网络时间同步）的影响，因此在长时间运行的程序中，时间值可能会发生变化。
    # time.perf_counter()：不受系统时间调整的影响，提供一个稳定的计时器，适合用于性能测量。
    # 简而言之，time.time()适合用于获取当前时间，而time.perf_counter()更适合用于高精度的时间间隔测量。
    start_time = time.perf_counter()

    # cdg:创建一个Flask.app应用
    app = create_flask_app_with_configs()
    # cdg:将Flask应用程序app传递给扩展模块的初始化方法，以便配置和准备扩展在应用中正常工作。
    initialize_extensions(app)
    end_time = time.perf_counter()
    if dify_config.DEBUG:
        logging.info(f"Finished create_app ({round((end_time - start_time) * 1000, 2)} ms)")
    return app


def initialize_extensions(app: DifyApp):
    # cdg:从extensions模块中导入各子模块配置，各子模块的内容及其组织方式都非常值得关注。
    # 将这些功能模块单独组织，然后在Flask应用程序中使用这些功能，增强应用的能力和灵活性。
    # 各模块的大概功能如下：
    # ext_app_metrics：用于应用程序性能监控。
    # ext_blueprints：用于组织Flask应用的蓝图。
    # ext_celery：用于集成Celery进行异步任务处理。
    # ext_code_based_extension：基于代码的扩展。
    # ext_commands：用于定义自定义命令。
    # ext_compress：用于压缩响应数据。
    # ext_database：用于数据库操作。
    # ext_hosting_provider：与远程托管服务相关的扩展。
    # ext_import_modules：用于动态导入模块。
    # ext_logging：用于日志记录。
    # ext_login：用于用户登录管理。
    # ext_mail：用于发送邮件。
    # ext_migrate：用于数据库迁移。
    # ext_proxy_fix：用于处理代理请求。
    # ext_redis：用于与 Redis 数据库交互。
    # ext_sentry：用于错误监控和报告。
    # ext_set_secretkey：用于设置应用的密钥。
    # ext_storage：用于文件存储。
    # ext_timezone：用于处理时区。
    # ext_warnings：可能用于处理警告信息。
    from extensions import (
        ext_app_metrics,
        ext_blueprints,
        ext_celery,
        ext_code_based_extension,
        ext_commands,
        ext_compress,
        ext_database,
        ext_hosting_provider,
        ext_import_modules,
        ext_logging,
        ext_login,
        ext_mail,
        ext_migrate,
        ext_proxy_fix,
        ext_redis,
        ext_sentry,
        ext_set_secretkey,
        ext_storage,
        ext_timezone,
        ext_warnings,
    )

    # cdg:将所有扩展子模块放到名为extensions列表中
    extensions = [
        ext_timezone,
        ext_logging,
        ext_warnings,
        ext_import_modules,
        ext_set_secretkey,
        ext_compress,
        ext_code_based_extension,
        ext_database,
        ext_app_metrics,
        ext_migrate,
        ext_redis,
        ext_storage,
        ext_celery,
        ext_login,
        ext_mail,
        ext_hosting_provider,
        ext_sentry,
        ext_proxy_fix,
        ext_blueprints,
        ext_commands,
    ]
    for ext in extensions:
        # cdg:获取扩展模块的简称
        short_name = ext.__name__.split(".")[-1]
        # cdg:检查每个模块对象是否具有is_enabled方法，如果有则调用该方法获取其返回值，否则默认将is_enabled设置为True
        is_enabled = ext.is_enabled() if hasattr(ext, "is_enabled") else True
        # cdg:如果未启用，而且在debug模式下，忽略该模块，不初始化到Flask应用程序中
        if not is_enabled:
            if dify_config.DEBUG:
                logging.info(f"Skipped {short_name}")
            continue

        start_time = time.perf_counter()
        # cdg:每个模块都有一个init_app的函数，接受一个参数app，其类型为DifyApp（也就是flask.Flask类型）
        # 用于初始化Flask应用程序的设置和扩展
        ext.init_app(app)
        end_time = time.perf_counter()
        # cdg:debug模式下才保存日志
        if dify_config.DEBUG:
            logging.info(f"Loaded {short_name} ({round((end_time - start_time) * 1000, 2)} ms)")


# cdg:定义一个名为create_migrations_app的函数，目的是创建并返回一个配置好的Flask应用实例，专门用于数据库迁移
def create_migrations_app():
    app = create_flask_app_with_configs()
    from extensions import ext_database, ext_migrate

    # Initialize only required extensions
    # cdg：仅初始化数据库操作和迁移操作两个必要的扩展模块
    ext_database.init_app(app)
    ext_migrate.init_app(app)

    return app
