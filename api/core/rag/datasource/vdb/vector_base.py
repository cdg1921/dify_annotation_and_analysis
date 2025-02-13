from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.rag.models.document import Document

# cdg:向量库供应商基础类，在BaseVector是一个抽象类，只定义了部分基础操作（函数），具体实现需在每个向量库供应商的类中实现
# cdg:例如：ChromaVector -> BaseVector，具体在api/core/rag/datasource/vdb路径下
class BaseVector(ABC):
    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    # cdg:获取向量库供应商类型
    @abstractmethod
    def get_type(self) -> str:
        raise NotImplementedError

    # cdg:创建向量库
    @abstractmethod
    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        raise NotImplementedError

    # cdg:向量库添加文本向量
    @abstractmethod
    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        raise NotImplementedError

    # cdg:判断向量库中文本段是否存在
    @abstractmethod
    def text_exists(self, id: str) -> bool:
        raise NotImplementedError

    # cdg:删除向量库中的文本段
    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        raise NotImplementedError

    # cdg:根据元数据字段获取向量库中的文本段ID
    def get_ids_by_metadata_field(self, key: str, value: str):
        raise NotImplementedError

    # cdg:根据元数据字段删除向量库中的文本段
    @abstractmethod
    def delete_by_metadata_field(self, key: str, value: str) -> None:
        raise NotImplementedError

    # cdg:利用向量搜索搜索向量库中的文本段
    @abstractmethod
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    # cdg:利用全文检索搜索向量库中的文本段
    @abstractmethod
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    # cdg:删除向量库
    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

    # cdg:过滤重复的文本段
    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            if text.metadata and "doc_id" in text.metadata:
                doc_id = text.metadata["doc_id"]
                exists_duplicate_node = self.text_exists(doc_id)
                if exists_duplicate_node:
                    texts.remove(text)

        return texts

    # cdg:获取文本段ID
    def _get_uuids(self, texts: list[Document]) -> list[str]:
        return [text.metadata["doc_id"] for text in texts if text.metadata and "doc_id" in text.metadata]

    # cdg:获取向量库名称
    @property
    def collection_name(self):
        return self._collection_name
