from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

# cdg: 定义SegmentUpdateEntity类，用于更新段落信息。
class SegmentUpdateEntity(BaseModel):
    content: str
    answer: Optional[str] = None
    keywords: Optional[list[str]] = None
    enabled: Optional[bool] = None

# cdg: 定义ParentMode类，用于表示父模式。
class ParentMode(str, Enum):
    FULL_DOC = "full-doc"
    PARAGRAPH = "paragraph"

# cdg: 定义NotionIcon类，用于表示Notion的图标。
class NotionIcon(BaseModel):
    type: str
    url: Optional[str] = None
    emoji: Optional[str] = None

# cdg: 定义NotionPage类，用于表示Notion的页面。
class NotionPage(BaseModel):
    page_id: str
    page_name: str
    page_icon: Optional[NotionIcon] = None
    type: str

# cdg: 定义NotionInfo类，用于表示Notion的信息。
class NotionInfo(BaseModel):
    workspace_id: str
    pages: list[NotionPage]

# cdg: 定义WebsiteInfo类，用于表示网站的信息。
class WebsiteInfo(BaseModel):
    provider: str
    job_id: str
    urls: list[str]
    only_main_content: bool = True

# cdg: 定义FileInfo类，用于表示文件的信息。
class FileInfo(BaseModel):
    file_ids: list[str]

# cdg: 定义InfoList类，用于表示信息列表。
class InfoList(BaseModel):
    data_source_type: Literal["upload_file", "notion_import", "website_crawl"]
    notion_info_list: Optional[list[NotionInfo]] = None
    file_info_list: Optional[FileInfo] = None
    website_info_list: Optional[WebsiteInfo] = None

# cdg: 定义DataSource类，用于表示数据源。
class DataSource(BaseModel):
    info_list: InfoList

# cdg: 定义PreProcessingRule类，用于表示预处理规则。
class PreProcessingRule(BaseModel):
    id: str
    enabled: bool

# cdg: 定义Segmentation类，用于表示分段。
class Segmentation(BaseModel):
    separator: str = "\n"
    max_tokens: int
    chunk_overlap: int = 0

# cdg: 定义Rule类，用于表示文本切分规则。
class Rule(BaseModel):
    pre_processing_rules: Optional[list[PreProcessingRule]] = None
    segmentation: Optional[Segmentation] = None
    parent_mode: Optional[Literal["full-doc", "paragraph"]] = None
    subchunk_segmentation: Optional[Segmentation] = None

# cdg: 定义ProcessRule类，用于表示文本切分规则。
class ProcessRule(BaseModel):
    mode: Literal["automatic", "custom", "hierarchical"]
    rules: Optional[Rule] = None

# cdg: 定义RerankingModel类，用于表示重排序模型。
class RerankingModel(BaseModel):
    reranking_provider_name: Optional[str] = None
    reranking_model_name: Optional[str] = None

# cdg: 定义RetrievalModel类，用于表示检索模型。
class RetrievalModel(BaseModel):
    search_method: Literal["hybrid_search", "semantic_search", "full_text_search"]
    reranking_enable: bool
    reranking_model: Optional[RerankingModel] = None
    top_k: int
    score_threshold_enabled: bool
    score_threshold: Optional[float] = None

# cdg: 定义KnowledgeConfig类，用于表示知识配置。
class KnowledgeConfig(BaseModel):
    original_document_id: Optional[str] = None
    duplicate: bool = True
    indexing_technique: Literal["high_quality", "economy"]
    data_source: DataSource
    process_rule: Optional[ProcessRule] = None
    retrieval_model: Optional[RetrievalModel] = None
    doc_form: str = "text_model"
    doc_language: str = "English"
    embedding_model: Optional[str] = None
    embedding_model_provider: Optional[str] = None
    name: Optional[str] = None

# cdg: 定义SegmentUpdateArgs类，用于表示段落更新参数。
class SegmentUpdateArgs(BaseModel):
    content: Optional[str] = None
    answer: Optional[str] = None
    keywords: Optional[list[str]] = None
    regenerate_child_chunks: bool = False
    enabled: Optional[bool] = None

# cdg: 定义ChildChunkUpdateArgs类，用于表示子段落更新参数。 
class ChildChunkUpdateArgs(BaseModel):
    id: Optional[str] = None
    content: str
