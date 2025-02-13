from abc import ABC, abstractmethod
from typing import Any, Optional

from configs import dify_config
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.cached_embedding import CacheEmbedding
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset, Whitelist

# cdg:AbstractVectorFactory类是一个抽象工厂，用于创建各种类型的向量存储。
class AbstractVectorFactory(ABC):
    @abstractmethod
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> BaseVector:
        raise NotImplementedError

    @staticmethod
    def gen_index_struct_dict(vector_type: VectorType, collection_name: str) -> dict:
        index_struct_dict = {"type": vector_type, "vector_store": {"class_prefix": collection_name}}
        return index_struct_dict


# cdg:在Vector类中，_init_vector方法用于初始化向量存储，根据指定的向量存储类型，选择对应的向量存储工厂，并调用其init_vector方法创建向量存储实例。
class Vector:
    def __init__(self, dataset: Dataset, attributes: Optional[list] = None):
        if attributes is None:
            attributes = ["doc_id", "dataset_id", "document_id", "doc_hash"]
        self._dataset = dataset
        self._embeddings = self._get_embeddings()
        self._attributes = attributes
        self._vector_processor = self._init_vector()

    def _init_vector(self) -> BaseVector:
        # cdg:向量库的类型在配置文件中由VECTOR_STORE变量指定
        vector_type = dify_config.VECTOR_STORE

        if self._dataset.index_struct_dict:
            vector_type = self._dataset.index_struct_dict["type"]
        else:

            if dify_config.VECTOR_STORE_WHITELIST_ENABLE:
                whitelist = (
                    db.session.query(Whitelist)
                    .filter(Whitelist.tenant_id == self._dataset.tenant_id, Whitelist.category == "vector_db")
                    .one_or_none()
                )
                if whitelist:
                    vector_type = VectorType.TIDB_ON_QDRANT

        if not vector_type:
            raise ValueError("Vector store must be specified.")

        vector_factory_cls = self.get_vector_factory(vector_type)
        return vector_factory_cls().init_vector(self._dataset, self._attributes, self._embeddings)

    # cdg:DIFY确定向量库供应商的方式与确定模型供应商的方式不一样，前者以match的方式指定，后者以读取本地配置文件的方式实现。
    @staticmethod
    def get_vector_factory(vector_type: str) -> type[AbstractVectorFactory]:
        # cdg:DIFY在确定向量库供应商时，采用match的方式，而不是读取本地配置文件。
        match vector_type:
            case VectorType.CHROMA:
                from core.rag.datasource.vdb.chroma.chroma_vector import ChromaVectorFactory

                return ChromaVectorFactory
            case VectorType.MILVUS:
                from core.rag.datasource.vdb.milvus.milvus_vector import MilvusVectorFactory

                return MilvusVectorFactory
            case VectorType.MYSCALE:
                from core.rag.datasource.vdb.myscale.myscale_vector import MyScaleVectorFactory

                return MyScaleVectorFactory
            case VectorType.PGVECTOR:
                from core.rag.datasource.vdb.pgvector.pgvector import PGVectorFactory

                return PGVectorFactory
            case VectorType.PGVECTO_RS:
                from core.rag.datasource.vdb.pgvecto_rs.pgvecto_rs import PGVectoRSFactory

                return PGVectoRSFactory
            case VectorType.QDRANT:
                from core.rag.datasource.vdb.qdrant.qdrant_vector import QdrantVectorFactory

                return QdrantVectorFactory
            case VectorType.RELYT:
                from core.rag.datasource.vdb.relyt.relyt_vector import RelytVectorFactory

                return RelytVectorFactory
            case VectorType.ELASTICSEARCH:
                from core.rag.datasource.vdb.elasticsearch.elasticsearch_vector import ElasticSearchVectorFactory

                return ElasticSearchVectorFactory
            case VectorType.ELASTICSEARCH_JA:
                from core.rag.datasource.vdb.elasticsearch.elasticsearch_ja_vector import (
                    ElasticSearchJaVectorFactory,
                )

                return ElasticSearchJaVectorFactory
            case VectorType.TIDB_VECTOR:
                from core.rag.datasource.vdb.tidb_vector.tidb_vector import TiDBVectorFactory

                return TiDBVectorFactory
            case VectorType.WEAVIATE:
                from core.rag.datasource.vdb.weaviate.weaviate_vector import WeaviateVectorFactory

                return WeaviateVectorFactory
            case VectorType.TENCENT:
                from core.rag.datasource.vdb.tencent.tencent_vector import TencentVectorFactory

                return TencentVectorFactory
            case VectorType.ORACLE:
                from core.rag.datasource.vdb.oracle.oraclevector import OracleVectorFactory

                return OracleVectorFactory
            case VectorType.OPENSEARCH:
                from core.rag.datasource.vdb.opensearch.opensearch_vector import OpenSearchVectorFactory

                return OpenSearchVectorFactory
            case VectorType.ANALYTICDB:
                from core.rag.datasource.vdb.analyticdb.analyticdb_vector import AnalyticdbVectorFactory

                return AnalyticdbVectorFactory
            case VectorType.COUCHBASE:
                from core.rag.datasource.vdb.couchbase.couchbase_vector import CouchbaseVectorFactory

                return CouchbaseVectorFactory
            case VectorType.BAIDU:
                from core.rag.datasource.vdb.baidu.baidu_vector import BaiduVectorFactory

                return BaiduVectorFactory
            case VectorType.VIKINGDB:
                from core.rag.datasource.vdb.vikingdb.vikingdb_vector import VikingDBVectorFactory

                return VikingDBVectorFactory
            case VectorType.UPSTASH:
                from core.rag.datasource.vdb.upstash.upstash_vector import UpstashVectorFactory

                return UpstashVectorFactory
            case VectorType.TIDB_ON_QDRANT:
                from core.rag.datasource.vdb.tidb_on_qdrant.tidb_on_qdrant_vector import TidbOnQdrantVectorFactory

                return TidbOnQdrantVectorFactory
            case VectorType.LINDORM:
                from core.rag.datasource.vdb.lindorm.lindorm_vector import LindormVectorStoreFactory

                return LindormVectorStoreFactory
            case VectorType.OCEANBASE:
                from core.rag.datasource.vdb.oceanbase.oceanbase_vector import OceanBaseVectorFactory

                return OceanBaseVectorFactory
            case _:
                raise ValueError(f"Vector store {vector_type} is not supported.")

    # cdg:以下函数具体实现详看CacheEmbedding和向量库供应商具体实例，其中self._embeddings是CacheEmbedding的实例，CacheEmbedding实现对document（文本段）和query的向量化
    # cdg:self._vector_processor详看get_vector_factory中对应的供应商实例
    def create(self, texts: Optional[list] = None, **kwargs):
        if texts:
            embeddings = self._embeddings.embed_documents([document.page_content for document in texts])
            # cdg:向量供应商的create函数，首先检查向量库（collection）是否存在，如果存在，则将Embedding数据入库，如果不存在，则先创建collection，并将collection的
            self._vector_processor.create(texts=texts, embeddings=embeddings, **kwargs)

    def add_texts(self, documents: list[Document], **kwargs):
        if kwargs.get("duplicate_check", False):
            documents = self._filter_duplicate_texts(documents)

        #cdg:将每个文本块向量化并加入到数据库中
        embeddings = self._embeddings.embed_documents([document.page_content for document in documents])
        self._vector_processor.create(texts=documents, embeddings=embeddings, **kwargs)

    # cdg:如果文本块已经存在，则不重复加入数据库
    def text_exists(self, id: str) -> bool:
        return self._vector_processor.text_exists(id)

    def delete_by_ids(self, ids: list[str]) -> None:
        self._vector_processor.delete_by_ids(ids)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        self._vector_processor.delete_by_metadata_field(key, value)

    def search_by_vector(self, query: str, **kwargs: Any) -> list[Document]:
        query_vector = self._embeddings.embed_query(query)
        return self._vector_processor.search_by_vector(query_vector, **kwargs)

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return self._vector_processor.search_by_full_text(query, **kwargs)

    def delete(self) -> None:
        self._vector_processor.delete()
        # delete collection redis cache
        if self._vector_processor.collection_name:
            collection_exist_cache_key = "vector_indexing_{}".format(self._vector_processor.collection_name)
            redis_client.delete(collection_exist_cache_key)

    def _get_embeddings(self) -> Embeddings:
        model_manager = ModelManager()

        embedding_model = model_manager.get_model_instance(
            tenant_id=self._dataset.tenant_id,
            provider=self._dataset.embedding_model_provider,
            model_type=ModelType.TEXT_EMBEDDING,
            model=self._dataset.embedding_model,
        )
        # cdg:返回CacheEmbedding实例，CacheEmbedding示例可以实现将document和query转为Embedding
        return CacheEmbedding(embedding_model)

    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            if text.metadata is None:
                continue
            doc_id = text.metadata["doc_id"]
            if doc_id:
                exists_duplicate_node = self.text_exists(doc_id)
                if exists_duplicate_node:
                    texts.remove(text)

        return texts

    def __getattr__(self, name):
        if self._vector_processor is not None:
            method = getattr(self._vector_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'vector_processor' object has no attribute '{name}'")
