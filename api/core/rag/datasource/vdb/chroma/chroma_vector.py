import json
from typing import Any, Optional

import chromadb
from chromadb import QueryResult, Settings
from pydantic import BaseModel

from configs import dify_config
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset


# cdg: ChromaConfig类用于配置ChromaDB客户端的参数，包括主机地址、端口号、租户、数据库名称、认证提供者、认证凭证等
class ChromaConfig(BaseModel):
    host: str
    port: int
    tenant: str
    database: str
    auth_provider: Optional[str] = None
    auth_credentials: Optional[str] = None

    def to_chroma_params(self):
        settings = Settings(
            # auth
            chroma_client_auth_provider=self.auth_provider,
            chroma_client_auth_credentials=self.auth_credentials,
        )

        return {
            "host": self.host,
            "port": self.port,
            "ssl": False,
            "tenant": self.tenant,
            "database": self.database,
            "settings": settings,
        }


# cdg: ChromaVector类继承自BaseVector类，并实现了一些特定的方法，如创建向量库、添加文本数据等
class ChromaVector(BaseVector):
    def __init__(self, collection_name: str, config: ChromaConfig):
        super().__init__(collection_name)
        self._client_config = config
        self._client = chromadb.HttpClient(**self._client_config.to_chroma_params())

    def get_type(self) -> str:
        return VectorType.CHROMA

    # cdg: ChromaVector类需要实现create方法，用于创建向量库，并调用add_texts方法添加文本数据到向量库中
    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        # cdg:首先根据collection_name获取到collection对象，然后调用collection.upsert方法，传入要添加的文本数据，即可创建向量库
        if texts:
            # create collection
            self.create_collection(self._collection_name)
            # cdg:添加文本数据到向量库中
            self.add_texts(texts, embeddings, **kwargs)

    # cdg: ChromaVector类需要实现create_collection方法，用于创建向量库，具体实现方式如下：
    def create_collection(self, collection_name: str):
        lock_name = "vector_indexing_lock_{}".format(collection_name)
        with redis_client.lock(lock_name, timeout=20):
            collection_exist_cache_key = "vector_indexing_{}".format(self._collection_name)
            if redis_client.get(collection_exist_cache_key):
                return
            # cdg:调用get_or_create_collection方法，传入collection_name，即可创建向量库
            self._client.get_or_create_collection(collection_name)
            # cdg:设置一个缓存，用于记录向量库是否存在，如果存在，则不再创建向量库,其中缓存的key为vector_indexing_{collection_name}，缓存的value为1，缓存的过期时间为3600秒
            redis_client.set(collection_exist_cache_key, 1, ex=3600)

    # cdg: ChromaVector类需要实现create方法，用于创建向量库，并调用add_texts方法添加文本数据到向量库中
    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        uuids = self._get_uuids(documents)
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        # cdg:首先根据collection_name获取到collection对象，然后调用collection.upsert方法，传入要添加的文本数据，即可创建向量库
        collection = self._client.get_or_create_collection(self._collection_name)
        # FIXME: chromadb using numpy array, fix the type error later
        collection.upsert(ids=uuids, documents=texts, embeddings=embeddings, metadatas=metadatas)  # type: ignore

    # cdg: ChromaVector类需要实现delete_by_metadata_field方法，用于根据元数据字段删除向量库中的文本数据
    def delete_by_metadata_field(self, key: str, value: str):
        collection = self._client.get_or_create_collection(self._collection_name)
        # FIXME: fix the type error later
        collection.delete(where={key: {"$eq": value}})  # type: ignore

    # cdg: ChromaVector类需要实现delete方法，用于删除向量库中的文本数据
    def delete(self):
        self._client.delete_collection(self._collection_name)

    # cdg: ChromaVector类需要实现delete_by_ids方法，用于根据ID列表删除向量库中的文本数据
    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        # cdg:首先根据collection_name获取到collection对象，然后调用collection.delete方法，传入要删除的id列表，即可删除向量库中的文本数据
        collection = self._client.get_or_create_collection(self._collection_name)
        collection.delete(ids=ids)

    # cdg: ChromaVector类需要实现text_exists方法，用于判断向量库中是否存在指定ID的文本数据
    def text_exists(self, id: str) -> bool:
        # cdg:首先根据collection_name获取到collection对象，然后调用collection.get方法，传入要查询的id，即可判断向量库中是否存在指定ID的文本数据
        collection = self._client.get_or_create_collection(self._collection_name)
        response = collection.get(ids=[id])
        return len(response) > 0

    # cdg: ChromaVector类需要实现search_by_vector方法，用于根据向量搜索向量库中的文本数据
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        # cdg:首先根据collection_name获取到collection对象，然后调用collection.query方法，传入要搜索的向量，即可搜索向量库中的文本数据
        collection = self._client.get_or_create_collection(self._collection_name)
        # cdg:返回搜索结果
        results: QueryResult = collection.query(query_embeddings=query_vector, n_results=kwargs.get("top_k", 4))
        # cdg:获取搜索结果的分数阈值，如果未设置，则默认为0
        score_threshold = float(kwargs.get("score_threshold") or 0.0)

        # cdg:如果搜索结果为空，则返回空列表
        # Check if results contain data
        if not results["ids"] or not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        docs = []
        for index in range(len(ids)):
            distance = distances[index]
            metadata = dict(metadatas[index])
            if distance >= score_threshold:
                metadata["score"] = distance
                doc = Document(
                    page_content=documents[index],
                    metadata=metadata,
                )
                docs.append(doc)
        # cdg:对搜索结果进行排序，按照分数从大到小排序
        # Sort the documents by score in descending order
        docs = sorted(docs, key=lambda x: x.metadata["score"] if x.metadata is not None else 0, reverse=True)
        return docs

    # cdg: ChromaVector目前不支持全文检索，所以返回空列表
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        # chroma does not support BM25 full text searching
        return []


# cdg:每个供应商除了需要基础BaseVector类实现独立的供应商类（如ChromaVector），还要相应地定义一个Factory类（如ChromaVectorFactory），以对外提供向量库服务
class ChromaVectorFactory(AbstractVectorFactory):
    # cdg:ChromaVector类需要实现init_vector方法，用于初始化向量库，并返回ChromaVector对象
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> BaseVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix.lower()
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id).lower()
            index_struct_dict = {"type": VectorType.CHROMA, "vector_store": {"class_prefix": collection_name}}
            dataset.index_struct = json.dumps(index_struct_dict)

        # cdg:返回ChromaVector对象
        return ChromaVector(
            collection_name=collection_name,
            config=ChromaConfig(
                host=dify_config.CHROMA_HOST or "",
                port=dify_config.CHROMA_PORT,
                tenant=dify_config.CHROMA_TENANT or chromadb.DEFAULT_TENANT,
                database=dify_config.CHROMA_DATABASE or chromadb.DEFAULT_DATABASE,
                auth_provider=dify_config.CHROMA_AUTH_PROVIDER,
                auth_credentials=dify_config.CHROMA_AUTH_CREDENTIALS,
            ),
        )
