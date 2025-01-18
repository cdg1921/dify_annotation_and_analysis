import os
import sys

# cdg:整个API服务入口代码
# cdg:源码启动API示例：flask run --host 0.0.0.0 --port=5001 --debug
# cdg:思考：为什么DIFY官方选择用Flask作为后端API框架，而不选择性能更加的Fast API?
# cdg:继续思考：为什么DIFY官方选择用python作为后端开发语言，而不选择Java或Go?
def is_db_command():
    if len(sys.argv) > 1 and sys.argv[0].endswith("flask") and sys.argv[1] == "db":
        return True
    return False

# create app
if is_db_command():  # cdg：默认为False
    from app_factory import create_migrations_app

    app = create_migrations_app()
else:
    # It seems that JetBrains Python debugger does not work well with gevent,
    # so we need to disable gevent in debug mode.
    # If you are using debugpy and set GEVENT_SUPPORT=True, you can debug with gevent.
    if (flask_debug := os.environ.get("FLASK_DEBUG", "0")) and flask_debug.lower() in {"false", "0", "no"}:
        from gevent import monkey  # type: ignore

        # gevent
        # cdg:在使用 gevent 库的时候，一般会在代码开头的地方执行 gevent.monkey.patch_all()，
        # 这行代码的作用是把标准库中的 socket 模块给替换掉，这样我们在使用 socket 的时候，
        # 不用修改任何代码就可以实现对代码的协程化，达到提升性能的目的。
        monkey.patch_all()

        from grpc.experimental import gevent as grpc_gevent  # type: ignore

        # grpc gevent
        grpc_gevent.init_gevent()

        import psycogreen.gevent  # type: ignore

        psycogreen.gevent.patch_psycopg()

    from app_factory import create_app

    app = create_app()
    celery = app.extensions["celery"]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
