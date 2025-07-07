from dify_app import DifyApp

# cdg: 使Flask应用支持数据库迁移功能
def init_app(app: DifyApp):
    # cdg: 在函数内部导入，避免在未用到迁移功能时强依赖该库。
    import flask_migrate  # type: ignore

    # cdg: SQLAlchemy的db实例，作为数据库操作的入口。
    from extensions.ext_database import db

    # cdg: 将Flask应用和数据库实例绑定到 Flask-Migrate，后续可通过命令行工具（如 flask db migrate/upgrade）进行数据库迁移操作。
    flask_migrate.Migrate(app, db)
