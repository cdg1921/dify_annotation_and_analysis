from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# cdg: Supabase存储配置信息，原为localStorage.Supabase是一种开源的分布式数据访问层，用于在不同存储系统之间提供统一的接口。
# Supabase的相比于其他存储系统，具有以下优势：
# 1. 支持多种存储系统，包括本地文件系统、S3、阿里云OSS、Azure Blob、百度云OBS、Google Cloud Storage、华为云OBS、OCI、腾讯云COS、火山引擎TOS、Supabase、本地存储等
# 2. 支持多种协议，包括HTTP、HTTPS、FTP、SFTP、WebDAV、S3、阿里云OSS、Azure Blob、百度云OBS、Google Cloud Storage、华为云OBS、OCI、腾讯云COS、火山引擎TOS、Supabase、本地存储等
# 3. 支持多种认证方式，包括API Key、OAuth、Basic Auth、Bearer Token、JWT等
# 4. 支持多种加密方式，包括AES、RSA、ECDSA、DSA、HMAC等
class SupabaseStorageConfig(BaseSettings):
    """
    Configuration settings for Supabase Object Storage Service
    """

    # cdg: 存储桶名称，默认值为None，当为None时，则不使用Supabase存储。此时，会使用localStorage.OpenDAL进行存储。
    SUPABASE_BUCKET_NAME: Optional[str] = Field(
        description="Name of the Supabase bucket to store and retrieve objects (e.g., 'dify-bucket')",
        default=None,
    )

    # cdg: API密钥，默认值为None
    SUPABASE_API_KEY: Optional[str] = Field(
        description="API KEY for authenticating with Supabase",
        default=None,
    )

    # cdg: 存储URL，默认值为None
    SUPABASE_URL: Optional[str] = Field(
        description="URL of the Supabase",
        default=None,
    )
