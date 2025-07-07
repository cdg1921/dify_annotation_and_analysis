import logging
from typing import Optional

from flask import Flask

from configs import dify_config
from dify_app import DifyApp

# cdg: 一个邮件发送扩展，主要用于在Flask应用中集成邮件发送功能。
class Mail:
    def __init__(self): # cdg: 初始化邮件客户端。
        self._client = None # cdg: 邮件客户端。
        self._default_send_from = None # cdg: 默认发送者。

    def is_inited(self) -> bool: # cdg: 检查邮件客户端是否初始化。
        return self._client is not None # cdg: 返回邮件客户端是否初始化。

    def init_app(self, app: Flask): # cdg: 初始化邮件客户端。
        mail_type = dify_config.MAIL_TYPE # cdg: 邮件类型。
        if not mail_type: # cdg: 如果邮件类型不存在。
            logging.warning("MAIL_TYPE is not set") # cdg: 日志记录。
            return # cdg: 返回。

        if dify_config.MAIL_DEFAULT_SEND_FROM:
            self._default_send_from = dify_config.MAIL_DEFAULT_SEND_FROM

        match mail_type: # cdg: 匹配邮件类型。
            case "resend": # cdg: 如果邮件类型为resend。
                import resend  # type: ignore # cdg: 导入resend。

                api_key = dify_config.RESEND_API_KEY # cdg: resend API密钥。
                if not api_key: # cdg: 如果resend API密钥不存在。
                    raise ValueError("RESEND_API_KEY is not set") # cdg: 抛出错误。

                api_url = dify_config.RESEND_API_URL # cdg: resend API URL。
                if api_url: # cdg: 如果resend API URL存在。
                    resend.api_url = api_url # cdg: 设置resend API URL。

                resend.api_key = api_key # cdg: 设置resend API密钥。
                self._client = resend.Emails # cdg: 设置邮件客户端。
            case "smtp": # cdg: 如果邮件类型为smtp。
                from libs.smtp import SMTPClient # cdg: 导入SMTPClient。

                if not dify_config.SMTP_SERVER or not dify_config.SMTP_PORT: # cdg: 如果SMTP服务器或端口不存在。
                    raise ValueError("SMTP_SERVER and SMTP_PORT are required for smtp mail type") # cdg: 抛出错误。
                if not dify_config.SMTP_USE_TLS and dify_config.SMTP_OPPORTUNISTIC_TLS: # cdg: 如果SMTP_USE_TLS和SMTP_OPPORTUNISTIC_TLS不存在。
                    raise ValueError("SMTP_OPPORTUNISTIC_TLS is not supported without enabling SMTP_USE_TLS") # cdg: 抛出错误。
                self._client = SMTPClient(
                    server=dify_config.SMTP_SERVER,
                    port=dify_config.SMTP_PORT,
                    username=dify_config.SMTP_USERNAME or "",
                    password=dify_config.SMTP_PASSWORD or "",
                    _from=dify_config.MAIL_DEFAULT_SEND_FROM or "",
                    use_tls=dify_config.SMTP_USE_TLS,
                    opportunistic_tls=dify_config.SMTP_OPPORTUNISTIC_TLS,
                )
            case _:
                raise ValueError("Unsupported mail type {}".format(mail_type))

    # cdg: 发送邮件
    def send(self, to: str, subject: str, html: str, from_: Optional[str] = None): # cdg: 发送邮件。
        if not self._client: # cdg: 如果邮件客户端不存在。
            raise ValueError("Mail client is not initialized") # cdg: 抛出错误。

        if not from_ and self._default_send_from: # cdg: 如果发送者不存在。
            from_ = self._default_send_from # cdg: 设置发送者。

        if not from_: # cdg: 如果发送者不存在。
            raise ValueError("mail from is not set") # cdg: 抛出错误。

        if not to: # cdg: 如果接收者不存在。
            raise ValueError("mail to is not set") # cdg: 抛出错误。

        if not subject: # cdg: 如果主题不存在。
            raise ValueError("mail subject is not set") # cdg: 抛出错误。

        if not html: # cdg: 如果HTML不存在。
            raise ValueError("mail html is not set") # cdg: 抛出错误。

        self._client.send(
            {
                "from": from_,
                "to": to,
                "subject": subject,
                "html": html,
            }
        )


def is_enabled() -> bool: # cdg: 检查邮件是否启用。
    return dify_config.MAIL_TYPE is not None and dify_config.MAIL_TYPE != "" # cdg: 返回邮件是否启用。

# cdg: 初始化邮件应用，这个函数的作用是初始化邮件应用，用于在Dify中使用邮件应用。
def init_app(app: DifyApp):
    mail.init_app(app)

# cdg: 创建邮件实例
mail = Mail()
