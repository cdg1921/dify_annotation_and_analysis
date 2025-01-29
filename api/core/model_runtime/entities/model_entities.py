from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from core.model_runtime.entities.common_entities import I18nObject


class ModelType(Enum):
    """
    Enum class for model type.
    """

    LLM = "llm"
    TEXT_EMBEDDING = "text-embedding"
    RERANK = "rerank"
    SPEECH2TEXT = "speech2text"
    MODERATION = "moderation"
    TTS = "tts"
    TEXT2IMG = "text2img"

    @classmethod
    def value_of(cls, origin_model_type: str) -> "ModelType":
        """
        Get model type from origin model type.

        :return: model type
        """
        if origin_model_type in {"text-generation", cls.LLM.value}:
            return cls.LLM
        elif origin_model_type in {"embeddings", cls.TEXT_EMBEDDING.value}:
            return cls.TEXT_EMBEDDING
        elif origin_model_type in {"reranking", cls.RERANK.value}:
            return cls.RERANK
        elif origin_model_type in {"speech2text", cls.SPEECH2TEXT.value}:
            return cls.SPEECH2TEXT
        elif origin_model_type in {"tts", cls.TTS.value}:
            return cls.TTS
        elif origin_model_type in {"text2img", cls.TEXT2IMG.value}:
            return cls.TEXT2IMG
        elif origin_model_type == cls.MODERATION.value:
            return cls.MODERATION
        else:
            raise ValueError(f"invalid origin model type {origin_model_type}")

    def to_origin_model_type(self) -> str:
        """
        Get origin model type from model type.

        :return: origin model type
        """
        if self == self.LLM:
            return "text-generation"
        elif self == self.TEXT_EMBEDDING:
            return "embeddings"
        elif self == self.RERANK:
            return "reranking"
        elif self == self.SPEECH2TEXT:
            return "speech2text"
        elif self == self.TTS:
            return "tts"
        elif self == self.MODERATION:
            return "moderation"
        elif self == self.TEXT2IMG:
            return "text2img"
        else:
            raise ValueError(f"invalid model type {self}")


class FetchFrom(Enum):
    """
    Enum class for fetch from.
    """

    PREDEFINED_MODEL = "predefined-model"         # cdg:预定义（内置）模型
    CUSTOMIZABLE_MODEL = "customizable-model"     # cdg:用户自定义模型


class ModelFeature(Enum):
    """
    Enum class for llm feature.
    """

    TOOL_CALL = "tool-call"
    MULTI_TOOL_CALL = "multi-tool-call"
    AGENT_THOUGHT = "agent-thought"
    VISION = "vision"
    STREAM_TOOL_CALL = "stream-tool-call"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"


class DefaultParameterName(StrEnum):
    """
    Enum class for parameter template variable.
    """

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOP_K = "top_k"
    # cdg:Presence Penalty（存在惩罚,取值范围：-2.0到2.0，默认为0）, 用于惩罚在生成文本中已经出现过的词汇。
    # 其目的是鼓励模型在生成过程中引入新的词汇或概念，避免重复。
    # 当presence_penalty的值增加时，模型会更倾向于使用尚未出现过的词汇，有助于提高生成文本的多样性和新颖性。
    # 适用于需要生成多样化内容的场景，例如创作、故事生成等。
    PRESENCE_PENALTY = "presence_penalty"
    # cdg:Frequency Penalty（频率惩罚,取值范围：-2.0到2.0，默认为0）,用于惩罚在生成文本中频繁出现的词汇。
    # 其目的是减少某些词汇的重复使用，尤其是那些在上下文中已经多次出现的词汇。
    # 当frequency_penalty的值增加时，模型会对已经出现过多次的词汇施加更大的惩罚，从而降低它们在后续生成中的出现概率，有助于避免生成文本中的冗余和重复。
    FREQUENCY_PENALTY = "frequency_penalty"
    # cdg:对比：
    # （1）presence_penalty关注的是词汇是否出现过（即是否存在），而frequency_penalty关注的是词汇出现的频率（即出现次数）。
    # （2）presence_penalty主要鼓励引入新词汇，而frequency_penalty则主要减少常用词汇的重复使用。
    # （3）presence_penalty更加关注文本的多样性，而frequency_penalty更加关注文本的流畅性和可读性。

    # cdg:max_tokens是模型输出的token长度，不包含输入
    MAX_TOKENS = "max_tokens"
    RESPONSE_FORMAT = "response_format"
    JSON_SCHEMA = "json_schema"

    @classmethod
    def value_of(cls, value: Any) -> "DefaultParameterName":
        """
        Get parameter name from value.

        :param value: parameter value
        :return: parameter name
        """
        for name in cls:
            if name.value == value:
                return name
        raise ValueError(f"invalid parameter name {value}")


class ParameterType(Enum):
    """
    Enum class for parameter type.
    """

    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOLEAN = "boolean"
    TEXT = "text"


class ModelPropertyKey(Enum):
    """
    Enum class for model property key.
    """

    MODE = "mode"
    CONTEXT_SIZE = "context_size"
    MAX_CHUNKS = "max_chunks"
    FILE_UPLOAD_LIMIT = "file_upload_limit"
    SUPPORTED_FILE_EXTENSIONS = "supported_file_extensions"
    MAX_CHARACTERS_PER_CHUNK = "max_characters_per_chunk"
    DEFAULT_VOICE = "default_voice"
    VOICES = "voices"
    WORD_LIMIT = "word_limit"
    AUDIO_TYPE = "audio_type"
    MAX_WORKERS = "max_workers"


# 供应商模型结构体
class ProviderModel(BaseModel):
    """
    Model class for provider model.
    """

    model: str
    label: I18nObject                                       # cdg:模型标识，用于展示，I18nObject支持便于切换系统语言
    model_type: ModelType                                   # cdg:LLM、Embedding、Reranker等
    features: Optional[list[ModelFeature]] = None           # cdg:模型特性，是否支持智能体、工具调用、视觉、语音等
    fetch_from: FetchFrom
    model_properties: dict[ModelPropertyKey, Any]
    deprecated: bool = False                                # cdg:是否废弃（不能使用）
    model_config = ConfigDict(protected_namespaces=())


# cdg:参数结构体
class ParameterRule(BaseModel):
    """
    Model class for parameter rule.
    """

    name: str
    use_template: Optional[str] = None
    label: I18nObject
    type: ParameterType
    help: Optional[I18nObject] = None
    required: bool = False
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    precision: Optional[int] = None
    options: list[str] = []


class PriceConfig(BaseModel):
    """
    Model class for pricing info.
    """

    input: Decimal                        # cdg:输入费用
    output: Optional[Decimal] = None      # cdg:输出费用
    unit: Decimal                         # cdg:计费单位，如token、千token等
    currency: str                         # 币种


class AIModelEntity(ProviderModel):
    """
    Model class for AI model.
    """
    # cdg:基础实体对象之一
    # cdg:AIModelEntity包含模型的各种参数信息及其计费方式，通用AIModel属性，还未涉及具体实例
    parameter_rules: list[ParameterRule] = []   # cdg:参数结构列表
    pricing: Optional[PriceConfig] = None       # cdg:计费方式


class ModelUsage(BaseModel):
    pass


class PriceType(Enum):
    """
    Enum class for price type.
    """

    INPUT = "input"
    OUTPUT = "output"


class PriceInfo(BaseModel):
    """
    Model class for price info.
    """

    unit_price: Decimal
    unit: Decimal
    total_amount: Decimal
    currency: str
