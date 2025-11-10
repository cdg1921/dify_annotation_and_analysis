import mimetypes
import os
import re
import urllib.parse
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel

from configs import dify_config

# cdg: 文件信息，用于存储文件信息，包括文件名、扩展名、MIME类型、大小等
class FileInfo(BaseModel):
    filename: str
    extension: str
    mimetype: str
    size: int

# cdg: 从响应中猜测文件信息，用于从响应中猜测文件信息，包括文件名、扩展名、MIME类型、大小等
def guess_file_info_from_response(response: httpx.Response):
    url = str(response.url)
    # Try to extract filename from URL
    # cdg: 解析URL，用于提取文件名
    parsed_url = urllib.parse.urlparse(url)
    url_path = parsed_url.path
    filename = os.path.basename(url_path)

    # cdg: 如果文件名无法提取，则使用Content-Disposition头部的文件名
    # If filename couldn't be extracted, use Content-Disposition header
    if not filename:
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename_match = re.search(r'filename="?(.+)"?', content_disposition)
            if filename_match:
                filename = filename_match.group(1)

    # cdg: 如果文件名仍然无法提取，则利用uuid生成一个唯一文件名
    # If still no filename, generate a unique one
    if not filename:
        unique_name = str(uuid4())
        filename = f"{unique_name}"

    # cdg: 猜测MIME类型，先从文件名猜测，如果失败则从URL猜测，如果仍然失败则从响应头部的Content-Type猜测
    # Guess MIME type from filename first, then URL
    mimetype, _ = mimetypes.guess_type(filename)
    if mimetype is None:
        mimetype, _ = mimetypes.guess_type(url)
    if mimetype is None:
        # If guessing fails, use Content-Type from response headers
        mimetype = response.headers.get("Content-Type", "application/octet-stream")

    # cdg: 获取文件扩展名，如果无法获取，则使用MIME类型猜测一个扩展名
    extension = os.path.splitext(filename)[1]

    # cdg: 如果文件扩展名无法获取，则使用MIME类型猜测一个扩展名
    # Ensure filename has an extension
    if not extension:
        extension = mimetypes.guess_extension(mimetype) or ".bin"
        filename = f"{filename}{extension}"

    # cdg: 返回文件信息，包括文件名、扩展名、MIME类型、大小
    return FileInfo(
        filename=filename,
        extension=extension,
        mimetype=mimetype,
        size=int(response.headers.get("Content-Length", -1)),
    )

# cdg: 从功能字典中获取功能特性配置，包括打开语句、建议问题、建议问题后回答、语音转文本、文本转语音、检索资源、注释回复、更多类似、用户输入表单、敏感词避免、文件上传等
def get_parameters_from_feature_dict(*, features_dict: Mapping[str, Any], user_input_form: list[dict[str, Any]]):
    return {
        "opening_statement": features_dict.get("opening_statement"),
        "suggested_questions": features_dict.get("suggested_questions", []),
        "suggested_questions_after_answer": features_dict.get("suggested_questions_after_answer", {"enabled": False}),
        "speech_to_text": features_dict.get("speech_to_text", {"enabled": False}),
        "text_to_speech": features_dict.get("text_to_speech", {"enabled": False}),
        "retriever_resource": features_dict.get("retriever_resource", {"enabled": False}),
        "annotation_reply": features_dict.get("annotation_reply", {"enabled": False}),
        "more_like_this": features_dict.get("more_like_this", {"enabled": False}),
        "user_input_form": user_input_form,
        "sensitive_word_avoidance": features_dict.get(
            "sensitive_word_avoidance", {"enabled": False, "type": "", "configs": []}
        ),
        "file_upload": features_dict.get(
            "file_upload",
            {
                "image": {
                    "enabled": False,
                    "number_limits": 3,
                    "detail": "high",
                    "transfer_methods": ["remote_url", "local_file"],
                }
            },
        ),
        "system_parameters": {
            "image_file_size_limit": dify_config.UPLOAD_IMAGE_FILE_SIZE_LIMIT,
            "video_file_size_limit": dify_config.UPLOAD_VIDEO_FILE_SIZE_LIMIT,
            "audio_file_size_limit": dify_config.UPLOAD_AUDIO_FILE_SIZE_LIMIT,
            "file_size_limit": dify_config.UPLOAD_FILE_SIZE_LIMIT,
            "workflow_file_upload_limit": dify_config.WORKFLOW_FILE_UPLOAD_LIMIT,
        },
    }
