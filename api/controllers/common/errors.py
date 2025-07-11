from werkzeug.exceptions import HTTPException

# cdg: 文件不存在错误
class FilenameNotExistsError(HTTPException):
    code = 400
    description = "The specified filename does not exist."

# cdg: 远程文件上传错误
class RemoteFileUploadError(HTTPException):
    code = 400
    description = "Error uploading remote file."
