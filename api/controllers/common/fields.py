from flask_restful import fields  # type: ignore

# cdg: 系统参数字段，用于规范化接口返回字段格式，一般和marshal_with装饰器配合使用
parameters__system_parameters = {
    "image_file_size_limit": fields.Integer,
    "video_file_size_limit": fields.Integer,
    "audio_file_size_limit": fields.Integer,
    "file_size_limit": fields.Integer,
    "workflow_file_upload_limit": fields.Integer,
}

# cdg: 参数字段，用于规范化接口返回字段格式，一般和marshal_with装饰器配合使用
parameters_fields = {
    "opening_statement": fields.String,
    "suggested_questions": fields.Raw,
    "suggested_questions_after_answer": fields.Raw,
    "speech_to_text": fields.Raw,
    "text_to_speech": fields.Raw,
    "retriever_resource": fields.Raw,
    "annotation_reply": fields.Raw,
    "more_like_this": fields.Raw,
    "user_input_form": fields.Raw,
    "sensitive_word_avoidance": fields.Raw,
    "file_upload": fields.Raw,
    "system_parameters": fields.Nested(parameters__system_parameters),
}
