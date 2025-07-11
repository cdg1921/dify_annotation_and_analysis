# cdg: 中间件管理配置信息，包括缓存、存储、向量数据库、数据库等配置信息
from typing import Any, Literal, Optional
from urllib.parse import quote_plus

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, computed_field
from pydantic_settings import BaseSettings

from .cache.redis_config import RedisConfig
from .storage.aliyun_oss_storage_config import AliyunOSSStorageConfig
from .storage.amazon_s3_storage_config import S3StorageConfig
from .storage.azure_blob_storage_config import AzureBlobStorageConfig
from .storage.baidu_obs_storage_config import BaiduOBSStorageConfig
from .storage.google_cloud_storage_config import GoogleCloudStorageConfig
from .storage.huawei_obs_storage_config import HuaweiCloudOBSStorageConfig
from .storage.oci_storage_config import OCIStorageConfig
from .storage.opendal_storage_config import OpenDALStorageConfig
from .storage.supabase_storage_config import SupabaseStorageConfig
from .storage.tencent_cos_storage_config import TencentCloudCOSStorageConfig
from .storage.volcengine_tos_storage_config import VolcengineTOSStorageConfig
from .vdb.analyticdb_config import AnalyticdbConfig
from .vdb.baidu_vector_config import BaiduVectorDBConfig
from .vdb.chroma_config import ChromaConfig
from .vdb.couchbase_config import CouchbaseConfig
from .vdb.elasticsearch_config import ElasticsearchConfig
from .vdb.lindorm_config import LindormConfig
from .vdb.milvus_config import MilvusConfig
from .vdb.myscale_config import MyScaleConfig
from .vdb.oceanbase_config import OceanBaseVectorConfig
from .vdb.opensearch_config import OpenSearchConfig
from .vdb.oracle_config import OracleConfig
from .vdb.pgvector_config import PGVectorConfig
from .vdb.pgvectors_config import PGVectoRSConfig
from .vdb.qdrant_config import QdrantConfig
from .vdb.relyt_config import RelytConfig
from .vdb.tencent_vector_config import TencentVectorDBConfig
from .vdb.tidb_on_qdrant_config import TidbOnQdrantConfig
from .vdb.tidb_vector_config import TiDBVectorConfig
from .vdb.upstash_config import UpstashConfig
from .vdb.vikingdb_config import VikingDBConfig
from .vdb.weaviate_config import WeaviateConfig

# cdg: 文件存储配置信息，包括Opendal、S3、阿里云OSS、Azure Blob、百度云OBS、Google Cloud Storage、华为云OBS、OCI、腾讯云COS、火山引擎TOS、Supabase、本地存储等
class StorageConfig(BaseSettings):
    STORAGE_TYPE: Literal[
        "opendal",
        "s3",
        "aliyun-oss",
        "azure-blob",
        "baidu-obs",
        "google-storage",
        "huawei-obs",
        "oci-storage",
        "tencent-cos",
        "volcengine-tos",
        "supabase",
        "local",
    ] = Field(
        description="Type of storage to use."
        " Options: 'opendal', '(deprecated) local', 's3', 'aliyun-oss', 'azure-blob', 'baidu-obs', 'google-storage', "
        "'huawei-obs', 'oci-storage', 'tencent-cos', 'volcengine-tos', 'supabase'. Default is 'opendal'.",
        default="opendal",
    )

    STORAGE_LOCAL_PATH: str = Field(
        description="Path for local storage when STORAGE_TYPE is set to 'local'.",
        default="storage",
        deprecated=True,
    )


# cdg: 向量数据库配置信息，包括Analyticdb、Chroma、Milvus、MyScale、OpenSearch、Oracle、PGVector、PGVectoRS、Qdrant、Relyt、TencentVectorDB、TiDBVector、Weaviate、Elasticsearch、Couchbase、VikingDB、Upstash、TidbOnQdrant、Lindorm、OceanBaseVector、BaiduVectorDB等
class VectorStoreConfig(BaseSettings):
    VECTOR_STORE: Optional[str] = Field(
        description="Type of vector store to use for efficient similarity search."
        " Set to None if not using a vector store.",
        default=None,
    )

    VECTOR_STORE_WHITELIST_ENABLE: Optional[bool] = Field(
        description="Enable whitelist for vector store.",
        default=False,
    )


# cdg: 关键词存储配置信息，包括Jieba、SnowNLP、HanLP、StanfordNLP、NLTK、CoreNLP、OpenNLP、Moses、MosesTokenizer、MosesTokenizer2、MosesTokenizer3、MosesTokenizer4、MosesTokenizer5、MosesTokenizer6、MosesTokenizer7、MosesTokenizer8、MosesTokenizer9、MosesTokenizer10等
class KeywordStoreConfig(BaseSettings):
    KEYWORD_STORE: str = Field(
        description="Method for keyword extraction and storage."
        " Default is 'jieba', a Chinese text segmentation library.",
        default="jieba",
    )


# cdg: 数据库配置信息，包括PostgreSQL、MySQL、SQLite、Oracle、SQL Server、MongoDB、Redis、Memcached、Elasticsearch、Solr、HBase、Cassandra、Couchbase、Neo4j、OrientDB、HyperSQL、H2、Derby、Firebird、Sybase、Informix、Ingres、Ingres-9.4、Ingres-10.0、Ingres-11.0、Ingres-12.0、Ingres-13.0、Ingres-14.0、Ingres-15.0、Ingres-16.0、Ingres-17.0、Ingres-18.0、Ingres-19.0、Ingres-20.0、Ingres-21.0、Ingres-22.0、Ingres-23.0、Ingres-24.0、Ingres-25.0、Ingres-26.0、Ingres-27.0、Ingres-28.0、Ingres-29.0、Ingres-30.0、Ingres-31.0、Ingres-32.0、Ingres-33.0、Ingres-34.0、Ingres-35.0、Ingres-36.0、Ingres-37.0、Ingres-38.0、Ingres-39.0、Ingres-40.0、Ingres-41.0、Ingres-42.0、Ingres-43.0、Ingres-44.0、Ingres-45.0、Ingres-46.0、Ingres-47.0、Ingres-48.0、Ingres-49.0、Ingres-50.0、Ingres-51.0、Ingres-52.0、Ingres-53.0、Ingres-54.0、Ingres-55.0、Ingres-56.0、Ingres-57.0、Ingres-58.0、Ingres-59.0、Ingres-60.0、Ingres-61.0、Ingres-62.0、Ingres-63.0、Ingres-64.0、Ingres-65.0、Ingres-66.0、Ingres-67.0、Ingres-68.0、Ingres-69.0、Ingres-70.0、Ingres-71.0、Ingres-72.0、Ingres-73.0、Ingres-74.0、Ingres-75.0、Ingres-76.0、Ingres-77.0、Ingres-78.0、Ingres-79.0、Ingres-80.0、
class DatabaseConfig(BaseSettings):
    DB_HOST: str = Field(
        description="Hostname or IP address of the database server.",
        default="localhost",
    )

    DB_PORT: PositiveInt = Field(
        description="Port number for database connection.",
        default=5432,
    )

    DB_USERNAME: str = Field(
        description="Username for database authentication.",
        default="postgres",
    )

    DB_PASSWORD: str = Field(
        description="Password for database authentication.",
        default="",
    )

    DB_DATABASE: str = Field(
        description="Name of the database to connect to.",
        default="dify",
    )

    DB_CHARSET: str = Field(
        description="Character set for database connection.",
        default="",
    )

    DB_EXTRAS: str = Field(
        description="Additional database connection parameters. Example: 'keepalives_idle=60&keepalives=1'",
        default="",
    )

    SQLALCHEMY_DATABASE_URI_SCHEME: str = Field(
        description="Database URI scheme for SQLAlchemy connection.",
        default="postgresql",
    )

    @computed_field
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        db_extras = (
            f"{self.DB_EXTRAS}&client_encoding={self.DB_CHARSET}" if self.DB_CHARSET else self.DB_EXTRAS
        ).strip("&")
        db_extras = f"?{db_extras}" if db_extras else ""
        return (
            f"{self.SQLALCHEMY_DATABASE_URI_SCHEME}://"
            f"{quote_plus(self.DB_USERNAME)}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_DATABASE}"
            f"{db_extras}"
        )

    SQLALCHEMY_POOL_SIZE: NonNegativeInt = Field(
        description="Maximum number of database connections in the pool.",
        default=30,
    )

    SQLALCHEMY_MAX_OVERFLOW: NonNegativeInt = Field(
        description="Maximum number of connections that can be created beyond the pool_size.",
        default=10,
    )

    SQLALCHEMY_POOL_RECYCLE: NonNegativeInt = Field(
        description="Number of seconds after which a connection is automatically recycled.",
        default=3600,
    )

    SQLALCHEMY_POOL_PRE_PING: bool = Field(
        description="If True, enables connection pool pre-ping feature to check connections.",
        default=False,
    )

    SQLALCHEMY_ECHO: bool | str = Field(
        description="If True, SQLAlchemy will log all SQL statements.",
        default=False,
    )

    @computed_field
    def SQLALCHEMY_ENGINE_OPTIONS(self) -> dict[str, Any]:
        return {
            "pool_size": self.SQLALCHEMY_POOL_SIZE,
            "max_overflow": self.SQLALCHEMY_MAX_OVERFLOW,
            "pool_recycle": self.SQLALCHEMY_POOL_RECYCLE,
            "pool_pre_ping": self.SQLALCHEMY_POOL_PRE_PING,
            "connect_args": {"options": "-c timezone=UTC"},
        }

# cdg: Celery配置信息，包括Celery后端、Celery消息队列、Celery使用Redis Sentinel等
class CeleryConfig(DatabaseConfig):
    # cdg: Celery后端，包括database、redis，默认使用database
    CELERY_BACKEND: str = Field(
        description="Backend for Celery task results. Options: 'database', 'redis'.",
        default="database",
    )
    # cdg: Celery消息队列，包括redis、rabbitmq，默认使用redis
    CELERY_BROKER_URL: Optional[str] = Field(
        description="URL of the message broker for Celery tasks.",
        default=None,
    )
    # cdg: 是否使用Redis Sentinel，默认使用False
    CELERY_USE_SENTINEL: Optional[bool] = Field(
        description="Whether to use Redis Sentinel for high availability.",
        default=False,
    )
    # cdg: Redis Sentinel主节点名称，默认使用None
    CELERY_SENTINEL_MASTER_NAME: Optional[str] = Field(
        description="Name of the Redis Sentinel master.",
        default=None,
    )
    # cdg: Redis Sentinel连接超时时间，默认使用0.1秒
    CELERY_SENTINEL_SOCKET_TIMEOUT: Optional[PositiveFloat] = Field(
        description="Timeout for Redis Sentinel socket operations in seconds.",
        default=0.1,
    )
    # cdg: Celery结果后端，包括database、redis，默认使用database
    @computed_field
    def CELERY_RESULT_BACKEND(self) -> str | None:
        return (
            "db+{}".format(self.SQLALCHEMY_DATABASE_URI)
            if self.CELERY_BACKEND == "database"
            else self.CELERY_BROKER_URL
        )
    # cdg: 是否使用SSL连接，默认使用False
    @property
    def BROKER_USE_SSL(self) -> bool:
        return self.CELERY_BROKER_URL.startswith("rediss://") if self.CELERY_BROKER_URL else False

# cdg: 内部测试配置信息，包括AWS Secret Access Key、AWS Access Key ID等
class InternalTestConfig(BaseSettings):
    """
    Configuration settings for Internal Test
    """

    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(
        description="Internal test AWS secret access key",
        default=None,
    )

    AWS_ACCESS_KEY_ID: Optional[str] = Field(
        description="Internal test AWS access key ID",
        default=None,
    )

# cdg: 中间件配置信息，通过类继承的方式集成了上述配置信息包括Celery、Database、KeywordStore、Redis、Storage、VectorStore、Analyticdb、Chroma、Milvus、MyScale、OpenSearch、Oracle、PGVector、PGVectoRS、Qdrant、Relyt、TencentVectorDB、TiDBVector、Weaviate、Elasticsearch、Couchbase、InternalTest、VikingDB、Upstash、TidbOnQdrant、Lindorm、OceanBaseVector、BaiduVectorDB等
class MiddlewareConfig(
    # place the configs in alphabet order
    CeleryConfig,
    DatabaseConfig,
    KeywordStoreConfig,
    RedisConfig,
    # configs of storage and storage providers
    StorageConfig,
    AliyunOSSStorageConfig,
    AzureBlobStorageConfig,
    BaiduOBSStorageConfig,
    GoogleCloudStorageConfig,
    HuaweiCloudOBSStorageConfig,
    OCIStorageConfig,
    OpenDALStorageConfig,
    S3StorageConfig,
    SupabaseStorageConfig,
    TencentCloudCOSStorageConfig,
    VolcengineTOSStorageConfig,
    # configs of vdb and vdb providers
    VectorStoreConfig,
    AnalyticdbConfig,
    ChromaConfig,
    MilvusConfig,
    MyScaleConfig,
    OpenSearchConfig,
    OracleConfig,
    PGVectorConfig,
    PGVectoRSConfig,
    QdrantConfig,
    RelytConfig,
    TencentVectorDBConfig,
    TiDBVectorConfig,
    WeaviateConfig,
    ElasticsearchConfig,
    CouchbaseConfig,
    InternalTestConfig,
    VikingDBConfig,
    UpstashConfig,
    TidbOnQdrantConfig,
    LindormConfig,
    OceanBaseVectorConfig,
    BaiduVectorDBConfig,
):
    pass
