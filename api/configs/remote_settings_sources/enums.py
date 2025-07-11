from enum import StrEnum

# cdg: 远程配置源名称枚举，包括Apollo、Nacos、Consul、Etcd、Zookeeper等，此处仅使用Apollo
class RemoteSettingsSourceName(StrEnum):
    APOLLO = "apollo"
