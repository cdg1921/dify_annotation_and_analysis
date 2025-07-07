from enum import StrEnum

# cdg: 存储类型，用于标识不同的存储类型，默认为local类型，旧版本中为local_storage,新版本中为opendal_storage
class StorageType(StrEnum):
    ALIYUN_OSS = "aliyun-oss"
    AZURE_BLOB = "azure-blob"
    BAIDU_OBS = "baidu-obs"
    GOOGLE_STORAGE = "google-storage"
    HUAWEI_OBS = "huawei-obs"
    LOCAL = "local"
    OCI_STORAGE = "oci-storage"
    OPENDAL = "opendal"
    S3 = "s3"
    TENCENT_COS = "tencent-cos"
    VOLCENGINE_TOS = "volcengine-tos"
    SUPBASE = "supabase"
