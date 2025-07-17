from enum import StrEnum

# cdg: 定义RecommendAppType枚举类，用于表示推荐应用类型。
class RecommendAppType(StrEnum):
    REMOTE = "remote"
    BUILDIN = "builtin"
    DATABASE = "db"
