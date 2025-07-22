import io
import logging
import uuid
from typing import Optional

from werkzeug.datastructures import FileStorage

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from models.model import App, AppMode, AppModelConfig, Message
from services.errors.audio import (
    AudioTooLargeServiceError,
    NoAudioUploadedServiceError,
    ProviderNotSupportSpeechToTextServiceError,
    ProviderNotSupportTextToSpeechServiceError,
    UnsupportedAudioTypeServiceError,
)

FILE_SIZE = 30
FILE_SIZE_LIMIT = FILE_SIZE * 1024 * 1024
ALLOWED_EXTENSIONS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "amr"]

logger = logging.getLogger(__name__)

# cdg: 音频服务，包括转录ASR、转录TTS、转录TTS语音。
class AudioService:

    # cdg: 转录ASR，将音频文件转换为文本。具体实现思路：
    # 1.检查应用模式，如果应用模式为高级聊天或工作流，则检查工作流是否启用ASR。
    # 2.如果应用模式为普通聊天，则检查应用模型配置是否启用ASR。
    # 3.检查文件是否为空，如果为空，则抛出NoAudioUploadedServiceError异常。
    # 4.检查文件扩展名是否为允许的扩展名，如果不允许，则抛出UnsupportedAudioTypeServiceError异常。
    # 5.检查文件大小是否超过限制。
    # 6.获取默认模型实例，如果模型实例为空，则抛出ProviderNotSupportSpeechToTextServiceError异常。
    # 7.通过Modelmanager获取默认ASR模型实例，调用ASR模型实例的invoke_speech2text方法，将音频文件转换为文本。
    # 8.返回文本。
    @classmethod
    def transcript_asr(cls, app_model: App, file: FileStorage, end_user: Optional[str] = None):
        if app_model.mode in {AppMode.ADVANCED_CHAT.value, AppMode.WORKFLOW.value}:
            workflow = app_model.workflow
            if workflow is None:
                raise ValueError("Speech to text is not enabled")

            features_dict = workflow.features_dict
            if "speech_to_text" not in features_dict or not features_dict["speech_to_text"].get("enabled"):
                raise ValueError("Speech to text is not enabled")
        else:
            app_model_config: AppModelConfig = app_model.app_model_config

            if not app_model_config.speech_to_text_dict["enabled"]:
                raise ValueError("Speech to text is not enabled")

        if file is None:
            raise NoAudioUploadedServiceError()

        extension = file.mimetype
        if extension not in [f"audio/{ext}" for ext in ALLOWED_EXTENSIONS]:
            raise UnsupportedAudioTypeServiceError()

        file_content = file.read()
        file_size = len(file_content)

        if file_size > FILE_SIZE_LIMIT:
            message = f"Audio size larger than {FILE_SIZE} mb"
            raise AudioTooLargeServiceError(message)

        model_manager = ModelManager()
        model_instance = model_manager.get_default_model_instance(
            tenant_id=app_model.tenant_id, model_type=ModelType.SPEECH2TEXT
        )
        if model_instance is None:
            raise ProviderNotSupportSpeechToTextServiceError()

        buffer = io.BytesIO(file_content)
        buffer.name = "temp.mp3"

        return {"text": model_instance.invoke_speech2text(file=buffer, user=end_user)}

    # cdg: 转录TTS，将文本转换为语音。具体实现思路：
    # 1.检查应用模式，如果应用模式为高级聊天或工作流，则检查工作流是否启用TTS。
    # 2.如果应用模式为普通聊天，则检查应用模型配置是否启用TTS。
    # 3.检查文本是否为空，如果为空，则抛出ValueError异常。
    # 4.检查语音是否为空，如果为空，则抛出ValueError异常。
    # 5.检查应用模型配置是否为空，如果为空，则抛出ValueError异常。
    # 6.检查应用模型配置是否启用TTS，如果未启用，则抛出ValueError异常。
    # 7.获取默认模型实例，如果模型实例为空，则抛出ProviderNotSupportTextToSpeechServiceError异常。
    # 8.通过Modelmanager获取默认TTS模型实例，调用TTS模型实例的invoke_tts方法，将文本转换为语音。
    # 9.返回语音。
    @classmethod
    def transcript_tts(
        cls,
        app_model: App,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        end_user: Optional[str] = None,
        message_id: Optional[str] = None,
    ):
        from collections.abc import Generator

        from flask import Response, stream_with_context

        from app import app
        from extensions.ext_database import db

        def invoke_tts(text_content: str, app_model: App, voice: Optional[str] = None):
            with app.app_context():
                if app_model.mode in {AppMode.ADVANCED_CHAT.value, AppMode.WORKFLOW.value}:
                    workflow = app_model.workflow
                    if workflow is None:
                        raise ValueError("TTS is not enabled")

                    features_dict = workflow.features_dict
                    if "text_to_speech" not in features_dict or not features_dict["text_to_speech"].get("enabled"):
                        raise ValueError("TTS is not enabled")

                    voice = features_dict["text_to_speech"].get("voice") if voice is None else voice
                else:
                    if app_model.app_model_config is None:
                        raise ValueError("AppModelConfig not found")
                    text_to_speech_dict = app_model.app_model_config.text_to_speech_dict

                    if not text_to_speech_dict.get("enabled"):
                        raise ValueError("TTS is not enabled")

                    voice = text_to_speech_dict.get("voice") if voice is None else voice

                model_manager = ModelManager()
                model_instance = model_manager.get_default_model_instance(
                    tenant_id=app_model.tenant_id, model_type=ModelType.TTS
                )
                try:
                    if not voice:
                        voices = model_instance.get_tts_voices()
                        if voices:
                            voice = voices[0].get("value")
                            if not voice:
                                raise ValueError("Sorry, no voice available.")
                        else:
                            raise ValueError("Sorry, no voice available.")

                    return model_instance.invoke_tts(
                        content_text=text_content.strip(), user=end_user, tenant_id=app_model.tenant_id, voice=voice
                    )
                except Exception as e:
                    raise e

        if message_id:
            try:
                uuid.UUID(message_id)
            except ValueError:
                return None
            message = db.session.query(Message).filter(Message.id == message_id).first()
            if message is None:
                return None
            if message.answer == "" and message.status == "normal":
                return None

            else:
                response = invoke_tts(message.answer, app_model=app_model, voice=voice)
                if isinstance(response, Generator):
                    return Response(stream_with_context(response), content_type="audio/mpeg")
                return response
        else:
            if text is None:
                raise ValueError("Text is required")
            response = invoke_tts(text, app_model, voice)
            if isinstance(response, Generator):
                return Response(stream_with_context(response), content_type="audio/mpeg")
            return response

    # cdg: 获取TTS语音列表。具体实现思路：
    # 1.获取默认模型实例，如果模型实例为空，则抛出ProviderNotSupportTextToSpeechServiceError异常。
    # 2.通过Modelmanager获取默认TTS模型实例，调用TTS模型实例的get_tts_voices方法，获取TTS语音列表。
    # 3.返回TTS语音列表。
    @classmethod
    def transcript_tts_voices(cls, tenant_id: str, language: str):
        model_manager = ModelManager()
        model_instance = model_manager.get_default_model_instance(tenant_id=tenant_id, model_type=ModelType.TTS)
        if model_instance is None:
            raise ProviderNotSupportTextToSpeechServiceError()

        try:
            return model_instance.get_tts_voices(language)
        except Exception as e:
            raise e
