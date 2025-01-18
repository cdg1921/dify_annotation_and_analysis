import os
import sys


# cdg:源码启动API示例：poetry run python -m flask run --host 0.0.0.0 --port=5001 --debug
# cdg:源码启动API示例（旧版本）：flask run --host 0.0.0.0 --port=5001 --debug
# cdg:思考：为什么DIFY官方选择用Flask作为后端API框架，而不选择性能更加的Fast API?
# cdg:继续思考：为什么DIFY官方选择用python作为后端开发语言，而不选择Java或Go?
def is_db_command():
    if len(sys.argv) > 1 and sys.argv[0].endswith("flask") and sys.argv[1] == "db":
        return True
    return False

# create app
# cdg:创建Flask App应用
if is_db_command():  # cdg：默认为False
    from app_factory import create_migrations_app

    app = create_migrations_app()
else:
    # It seems that JetBrains Python debugger does not work well with gevent,
    # so we need to disable gevent in debug mode.
    # If you are using debugpy and set GEVENT_SUPPORT=True, you can debug with gevent.
    # cdg:非debug模式下
    if (flask_debug := os.environ.get("FLASK_DEBUG", "0")) and flask_debug.lower() in {"false", "0", "no"}:
        from gevent import monkey  # type: ignore

        # gevent
        # cdg:在使用 gevent 库的时候，一般会在代码开头的地方执行gevent.monkey.patch_all()，
        # 这行代码的作用是把标准库中的socket模块给替换掉，这样我们在使用socket的时候，
        # 不用修改任何代码就可以实现对代码的协程化，达到提升性能的目的。
        monkey.patch_all()

        # cdg:“type: ignore”提示类型检查时忽略这一行的错误，用于在使用某些不完全符合类型提示的库时，
        # 避免类型检查器发出警告。
        from grpc.experimental import gevent as grpc_gevent  # type: ignore

        # grpc gevent
        grpc_gevent.init_gevent()
        # cdg:初始化gevent，使得gRPC可以在gevent的协程环境中运行。
        # gevent是一个基于协程的Python网络库，能够实现高并发的网络应用。
        # 通过初始化，gRPC可以利用gevent的异步特性来处理并发请求，提高性能。

        # cdg:psycogreen是一个用于将psycopg（一个流行的PostgreSQL数据库适配器）与gevent协程库结合使用的库。
        import psycogreen.gevent  # type: ignore

        # cdg:以下一行代码的作用是对psycopg进行补丁处理，使其能够与gevent 协同工作。通过打补丁，psycopg 的阻塞操作（如数据库查询）将被转换为非阻塞操作，从而允许其他协程在等待数据库响应时继续执行。这使得在使用 gevent 的应用程序中，数据库操作不会阻塞整个事件循环，从而提高了并发性能。
        psycogreen.gevent.patch_psycopg()

    from app_factory import create_app

    app = create_app()
    celery = app.extensions["celery"]

if __name__ == "__main__":
    # cdg:整个API服务入口代码，启动Flask APP服务
    app.run(host="0.0.0.0", port=5001)
