from pydantic import Field
from pydantic_settings import BaseSettings

# cdg: Opendal（Open Data Access Layer）存储配置信息，原为localStorage.OpenDAL是一种开源的分布式数据访问层，用于在不同存储系统之间提供统一的接口。
# Opendal的相比于其他存储系统，具有以下优势：
# 1. 支持多种存储系统，包括本地文件系统、S3、阿里云OSS、Azure Blob、百度云OBS、Google Cloud Storage、华为云OBS、OCI、腾讯云COS、火山引擎TOS、Supabase、本地存储等
# 2. 支持多种协议，包括HTTP、HTTPS、FTP、SFTP、WebDAV、S3、阿里云OSS、Azure Blob、百度云OBS、Google Cloud Storage、华为云OBS、OCI、腾讯云COS、火山引擎TOS、Supabase、本地存储等
# 3. 支持多种认证方式，包括API Key、OAuth、Basic Auth、Bearer Token、JWT等
# 4. 支持多种加密方式，包括AES、RSA、ECDSA、DSA、HMAC等
# 5. 支持多种压缩方式，包括GZIP、BZIP2、XZ、ZSTD等
# 6. 支持多种分片方式，包括Range、Chunk、Block等
# 7. 支持多种缓存方式，包括Memory、Redis、Memcached、Disk等
# 8. 支持多种日志方式，包括Console、File、Sentry、Datadog等
# 9. 支持多种监控方式，包括Prometheus、Grafana、Datadog等
# 10. 支持多种告警方式，包括Email、SMS、Webhook等
# 11. 支持多种备份方式，包括Snapshot、Replication、Backup等
# 12. 支持多种恢复方式，包括Restore、Recover、Redo等
class OpenDALStorageConfig(BaseSettings):
    OPENDAL_SCHEME: str = Field( # cdg: 默认使用fs协议（filesystem），即本地文件系统
        default="fs",
        description="OpenDAL scheme.",
    )
