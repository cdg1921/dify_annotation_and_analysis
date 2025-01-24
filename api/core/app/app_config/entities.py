from collections.abc import Sequence
from enum import Enum, StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from core.file import FileTransferMethod, FileType, FileUploadConfig
from core.model_runtime.entities.message_entities import PromptMessageRole
from models.model import AppMode


class ModelConfigEntity(BaseModel):
    """
    Model Config Entity.
    """

    provider: str                     # cdg:供应商
    model: str                        # cdg:模型名称
    mode: Optional[str] = None        # cdg:模型模式：chat或completion
    parameters: dict[str, Any] = {}   # cdg:模型参数，如max_tokens、top K、Top P、温度等
    stop: list[str] = []              # cdg:停止标识


class AdvancedChatMessageEntity(BaseModel):
    """
    Advanced Chat Message Entity.
    """

    text: str                  # cdg:消息内容
    role: PromptMessageRole    # cdg:角色，如system、user、assistant、tool等


class AdvancedChatPromptTemplateEntity(BaseModel):
    """
    Advanced Chat Prompt Template Entity.
    """

    messages: list[AdvancedChatMessageEntity]   # cdg:由多个AdvancedChatMessageEntity构成的消息列表


class AdvancedCompletionPromptTemplateEntity(BaseModel):
    """
    Advanced Completion Prompt Template Entity.
    """

    class RolePrefixEntity(BaseModel):
        """
        Role Prefix Entity.
        """
        # cdg:Completion模式下的角色只有user和assistant
        user: str
        assistant: str

    prompt: str
    role_prefix: Optional[RolePrefixEntity] = None


class PromptTemplateEntity(BaseModel):
    """
    Prompt Template Entity.
    """

    class PromptType(Enum):
        """
        Prompt Type.
        'simple', 'advanced'
        """

        SIMPLE = "simple"
        ADVANCED = "advanced"

        @classmethod
        def value_of(cls, value: str):
            """
            Get value of given mode.

            :param value: mode value
            :return: mode
            """
            for mode in cls:
                if mode.value == value:
                    return mode
            raise ValueError(f"invalid prompt type value {value}")

    prompt_type: PromptType
    simple_prompt_template: Optional[str] = None   # cdg:Optional,可选，可有可无
    advanced_chat_prompt_template: Optional[AdvancedChatPromptTemplateEntity] = None
    advanced_completion_prompt_template: Optional[AdvancedCompletionPromptTemplateEntity] = None


class VariableEntityType(StrEnum):
    TEXT_INPUT = "text-input"
    SELECT = "select"
    PARAGRAPH = "paragraph"
    NUMBER = "number"
    EXTERNAL_DATA_TOOL = "external_data_tool"
    FILE = "file"
    FILE_LIST = "file-list"


class VariableEntity(BaseModel):
    """
    Variable Entity.
    """

    variable: str
    label: str
    description: str = ""
    type: VariableEntityType
    required: bool = False
    max_length: Optional[int] = None
    options: Sequence[str] = Field(default_factory=list)
    allowed_file_types: Sequence[FileType] = Field(default_factory=list)
    allowed_file_extensions: Sequence[str] = Field(default_factory=list)
    allowed_file_upload_methods: Sequence[FileTransferMethod] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def convert_none_description(cls, v: Any) -> str:
        return v or ""

    @field_validator("options", mode="before")
    @classmethod
    def convert_none_options(cls, v: Any) -> Sequence[str]:
        return v or []


class ExternalDataVariableEntity(BaseModel):
    """
    External Data Variable Entity.
    """

    variable: str
    type: str
    config: dict[str, Any] = {}


class DatasetRetrieveConfigEntity(BaseModel):
    """
    Dataset Retrieve Config Entity.
    """

    class RetrieveStrategy(Enum):
        """
        Dataset Retrieve Strategy.
        'single' or 'multiple'
        """

        SINGLE = "single"       # cdg:仅召回单个知识库的内容
        MULTIPLE = "multiple"   # cdg:同时召回多个知识库的内容

        @classmethod
        def value_of(cls, value: str):
            """
            Get value of given mode.

            :param value: mode value
            :return: mode
            """
            for mode in cls:
                if mode.value == value:
                    return mode
            raise ValueError(f"invalid retrieve strategy value {value}")

    query_variable: Optional[str] = None  # Only when app mode is completion

    retrieve_strategy: RetrieveStrategy
    top_k: Optional[int] = None
    score_threshold: Optional[float] = 0.0
    rerank_mode: Optional[str] = "reranking_model"
    reranking_model: Optional[dict] = None
    weights: Optional[dict] = None
    reranking_enabled: Optional[bool] = True


class DatasetEntity(BaseModel):
    """
    Dataset Config Entity.
    """

    dataset_ids: list[str]
    retrieve_config: DatasetRetrieveConfigEntity


class SensitiveWordAvoidanceEntity(BaseModel):
    """
    Sensitive Word Avoidance Entity.
    """
    # cdg:敏感词字典
    type: str
    config: dict[str, Any] = {}


class TextToSpeechEntity(BaseModel):
    """
    Sensitive Word Avoidance Entity.
    """

    enabled: bool
    voice: Optional[str] = None
    language: Optional[str] = None


class TracingConfigEntity(BaseModel):
    """
    Tracing Config Entity.
    """

    enabled: bool
    tracing_provider: str

# cdg:工作流中“添加功能”模块支持的功能特性
class AppAdditionalFeatures(BaseModel):
    file_upload: Optional[FileUploadConfig] = None
    opening_statement: Optional[str] = None
    suggested_questions: list[str] = []
    suggested_questions_after_answer: bool = False
    show_retrieve_source: bool = False
    more_like_this: bool = False
    speech_to_text: bool = False
    text_to_speech: Optional[TextToSpeechEntity] = None
    trace_config: Optional[TracingConfigEntity] = None


class AppConfig(BaseModel):
    """
    Application Config Entity.
    """

    tenant_id: str
    app_id: str
    app_mode: AppMode
    # cdg:支持completion、workflow、chat、advanced-chat、agent-chat、channel
    additional_features: AppAdditionalFeatures
    variables: list[VariableEntity] = []
    sensitive_word_avoidance: Optional[SensitiveWordAvoidanceEntity] = None


class EasyUIBasedAppModelConfigFrom(Enum):
    """
    App Model Config From.
    """

    ARGS = "args"
    APP_LATEST_CONFIG = "app-latest-config"
    CONVERSATION_SPECIFIC_CONFIG = "conversation-specific-config"


class EasyUIBasedAppConfig(AppConfig):
    """
    Easy UI Based App Config Entity.
    """

    app_model_config_from: EasyUIBasedAppModelConfigFrom
    app_model_config_id: str
    app_model_config_dict: dict
    model: ModelConfigEntity
    prompt_template: PromptTemplateEntity
    dataset: Optional[DatasetEntity] = None
    external_data_variables: list[ExternalDataVariableEntity] = []


class WorkflowUIBasedAppConfig(AppConfig):
    """
    Workflow UI Based App Config Entity.
    """

    workflow_id: str
