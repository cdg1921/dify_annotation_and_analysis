from libs.exception import BaseHTTPException

# cdg:已安装错误，提示已安装成功，请刷新页面或返回首页。
class AlreadySetupError(BaseHTTPException):
    error_code = "already_setup"
    description = "Dify has been successfully installed. Please refresh the page or return to the dashboard homepage."
    code = 403

# cdg:未安装错误，提示未安装成功，请先安装。
class NotSetupError(BaseHTTPException):
    error_code = "not_setup"
    description = (
        "Dify has not been initialized and installed yet. "
        "Please proceed with the initialization and installation process first."
    )
    code = 401

# cdg:未初始化验证错误，提示未初始化验证成功，请先初始化验证。
class NotInitValidateError(BaseHTTPException):
    error_code = "not_init_validated"
    description = "Init validation has not been completed yet. Please proceed with the init validation process first."
    code = 401

# cdg:初始化验证失败错误，提示初始化验证失败，请检查密码并重试。
class InitValidateFailedError(BaseHTTPException):
    error_code = "init_validate_failed"
    description = "Init validation failed. Please check the password and try again."
    code = 401

# cdg:账号未关联租户错误，提示账号未关联租户。
class AccountNotLinkTenantError(BaseHTTPException):
    error_code = "account_not_link_tenant"
    description = "Account not link tenant."
    code = 403

# cdg:账号已激活错误，提示账号已激活，请检查密码并重试。
class AlreadyActivateError(BaseHTTPException):
    error_code = "already_activate"
    description = "Auth Token is invalid or account already activated, please check again."
    code = 403

# cdg:不允许创建工作区错误，提示工作区不存在，请联系管理员邀请加入工作区。
class NotAllowedCreateWorkspace(BaseHTTPException):
    error_code = "not_allowed_create_workspace"
    description = "Workspace not found, please contact system admin to invite you to join in a workspace."
    code = 400

# cdg:账号被封禁错误，提示账号被封禁。
class AccountBannedError(BaseHTTPException):
    error_code = "account_banned"
    description = "Account is banned."
    code = 400

# cdg:账号不存在错误，提示账号不存在。
class AccountNotFound(BaseHTTPException):
    error_code = "account_not_found"
    description = "Account not found."
    code = 400

# cdg:邮箱发送IP限制错误，提示发送邮箱过多，请稍后再试。
class EmailSendIpLimitError(BaseHTTPException):
    error_code = "email_send_ip_limit"
    description = "Too many emails have been sent from this IP address recently. Please try again later."
    code = 429

# cdg:文件太大错误，提示文件太大，请检查文件大小。
class FileTooLargeError(BaseHTTPException):
    error_code = "file_too_large"
    description = "File size exceeded. {message}"
    code = 413

# cdg:不支持的文件类型错误，提示文件类型不支持。
class UnsupportedFileTypeError(BaseHTTPException):
    error_code = "unsupported_file_type"
    description = "File type not allowed."
    code = 415

# cdg:文件太多错误，提示只能上传一个文件。
class TooManyFilesError(BaseHTTPException):
    error_code = "too_many_files"
    description = "Only one file is allowed."
    code = 400

# cdg:没有上传文件错误，提示请上传文件。
class NoFileUploadedError(BaseHTTPException):
    error_code = "no_file_uploaded"
    description = "Please upload your file."
    code = 400

# cdg:未授权和强制退出错误，提示未授权和强制退出。
class UnauthorizedAndForceLogout(BaseHTTPException):
    error_code = "unauthorized_and_force_logout"
    description = "Unauthorized and force logout."
    code = 401

# cdg:账号冻结错误，提示账号冻结。
class AccountInFreezeError(BaseHTTPException):
    error_code = "account_in_freeze"
    code = 400
    description = (
        "This email account has been deleted within the past 30 days"
        "and is temporarily unavailable for new account registration."
    )
