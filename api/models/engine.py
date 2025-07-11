from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData

# cdg: 定义数据库索引命名规范
POSTGRES_INDEXES_NAMING_CONVENTION = {
    "ix": "%(column_0_label)s_idx", # cdg: 索引名称为"列名_idx"
    "uq": "%(table_name)s_%(column_0_name)s_key", # cdg: 唯一约束名称为"表名_列名_key"    
    "ck": "%(table_name)s_%(constraint_name)s_check", # cdg: 检查约束名称为"表名_约束名_check"
    "fk": "%(table_name)s_%(column_0_name)s_fkey", # cdg: 外键约束名称为"表名_列名_fkey"
    "pk": "%(table_name)s_pkey", # cdg: 主键约束名称为"表名_pkey" 
}

# cdg: 创建数据库引擎，DIFY使用SQLAlchemy作为ORM框架，SQLAlchemy是一个强大的ORM框架，支持多种数据库，包括PostgreSQL、MySQL、SQLite等，国产化方面支持达梦数据库。
metadata = MetaData(naming_convention=POSTGRES_INDEXES_NAMING_CONVENTION)
db = SQLAlchemy(metadata=metadata)
