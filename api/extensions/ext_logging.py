import logging
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler

import flask

from configs import dify_config # cdg: 自定义配置，包含日志相关参数。
from dify_app import DifyApp    # cdg: 自定义 Flask 应用对象。

# cdg: 初始化初始化和配置日志系统的模块，主要用于Flask应用的日志记录。
def init_app(app: DifyApp):
    log_handlers: list[logging.Handler] = [] # cdg: 日志处理程序列表。
    log_file = dify_config.LOG_FILE # cdg: 日志文件路径。
    if log_file:
        log_dir = os.path.dirname(log_file) # cdg: 日志目录。
        os.makedirs(log_dir, exist_ok=True) # cdg: 创建日志目录。
        log_handlers.append(
            RotatingFileHandler(
                filename=log_file, # cdg: 日志文件路径。
                maxBytes=dify_config.LOG_FILE_MAX_SIZE * 1024 * 1024, # cdg: 日志文件最大大小。
                backupCount=dify_config.LOG_FILE_BACKUP_COUNT, # cdg: 日志文件备份数量。
            )
        )

    # Always add StreamHandler to log to console
    sh = logging.StreamHandler(sys.stdout) # cdg: 标准输出流处理器。
    sh.addFilter(RequestIdFilter()) # cdg: 添加请求ID过滤器。
    log_formatter = logging.Formatter(fmt=dify_config.LOG_FORMAT) # cdg: 日志格式化器。
    sh.setFormatter(log_formatter) # cdg: 设置日志格式化器。
    log_handlers.append(sh) # cdg: 添加日志处理程序。

    logging.basicConfig(
        level=dify_config.LOG_LEVEL, # cdg: 日志级别。
        datefmt=dify_config.LOG_DATEFORMAT, # cdg: 日志日期格式。
        handlers=log_handlers, # cdg: 日志处理程序。
        force=True, # cdg: 强制配置。
    )
    log_tz = dify_config.LOG_TZ # cdg: 日志时区。
    if log_tz:
        from datetime import datetime # cdg: 日期时间模块。

        import pytz # cdg: 时区模块。

        timezone = pytz.timezone(log_tz) # cdg: 时区。

        def time_converter(seconds): # cdg: 时间转换器。
            return datetime.fromtimestamp(seconds, tz=timezone).timetuple() # cdg: 时间转换器。

        for handler in logging.root.handlers:
            if handler.formatter: # cdg: 如果处理器有格式化器。
                handler.formatter.converter = time_converter # cdg: 设置时间转换器。


def get_request_id(): # cdg: 获取请求ID。
    if getattr(flask.g, "request_id", None): # cdg: 如果请求ID存在。
        return flask.g.request_id # cdg: 返回请求ID。

    new_uuid = uuid.uuid4().hex[:10] # cdg: 生成请求ID。
    flask.g.request_id = new_uuid # cdg: 设置请求ID。

    return new_uuid # cdg: 返回请求ID。


class RequestIdFilter(logging.Filter): # cdg: 请求ID过滤器。
    # This is a logging filter that makes the request ID available for use in
    # the logging format. Note that we're checking if we're in a request
    # context, as we may want to log things before Flask is fully loaded.
    def filter(self, record): # cdg: 过滤器。
        record.req_id = get_request_id() if flask.has_request_context() else "" # cdg: 设置请求ID。
        return True # cdg: 返回True。
