import os
import time

from dify_app import DifyApp

# cdg: 初始化时区
def init_app(app: DifyApp):
    os.environ["TZ"] = "UTC" # cdg: 设置时区为UTC，UTC是协调世界时，是国际标准时间。
    # windows platform not support tzset
    if hasattr(time, "tzset"): # cdg: 如果系统支持tzset，则设置时区。
        time.tzset()
    # cdg: UTC（Coodinated Universal Time），协调世界时，又称世界统一时间、世界标准时间、国际协调时间。由于英文（CUT）和法文（TUC）的缩写不同，作为妥协，简称UTC。
    # UTC是现在全球通用的时间标准，全球各地都同意将各自的时间进行同步协调。UTC时间是经过平均太阳时（以格林威治时间GMT为准）、地轴运动修正后的新时标以及以秒为单位的国际原子时所综合精算而成。
