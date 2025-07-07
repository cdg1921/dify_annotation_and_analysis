from typing import Any, Union

import redis
from redis.cluster import ClusterNode, RedisCluster
from redis.connection import Connection, SSLConnection
from redis.sentinel import Sentinel

from configs import dify_config
from dify_app import DifyApp

# cdg: 
class RedisClientWrapper:
    """
    A wrapper class for the Redis client that addresses the issue where the global
    `redis_client` variable cannot be updated when a new Redis instance is returned
    by Sentinel.

    This class allows for deferred initialization of the Redis client, enabling the
    client to be re-initialized with a new instance when necessary. This is particularly
    useful in scenarios where the Redis instance may change dynamically, such as during
    a failover in a Sentinel-managed Redis setup.

    Attributes:
        _client (redis.Redis): The actual Redis client instance. It remains None until
                               initialized with the `initialize` method.

    Methods:
        initialize(client): Initializes the Redis client if it hasn't been initialized already.
        __getattr__(item): Delegates attribute access to the Redis client, raising an error
                           if the client is not initialized.
    """
    """cdg:
    一个用于 Redis客户端的包装类，旨在解决当Sentinel返回新的Redis实例时，全局变量 `redis_client` 无法被更新的问题。
    该类允许对Redis客户端进行延迟初始化，使得在有必要时可以用新的实例重新初始化客户端。这在Redis实例可能动态变化（如在由Sentinel管理的Redis集群发生故障转移时）等场景下尤其有用。

    属性：
        _client (redis.Redis)：实际的Redis客户端实例。在通过 `initialize` 方法初始化之前，它保持为 None。

    方法：
        initialize(client)：如果尚未初始化Redis客户端，则进行初始化。
        __getattr__(item)：将属性访问委托给Redis客户端，如果客户端尚未初始化则抛出错误。
    """

    def __init__(self):
        self._client = None

    def initialize(self, client):
        if self._client is None:
            self._client = client

    def __getattr__(self, item):
        if self._client is None:
            raise RuntimeError("Redis client is not initialized. Call init_app first.")
        return getattr(self._client, item)


redis_client = RedisClientWrapper()

# cdg: 初始化Redis客户端
def init_app(app: DifyApp):
    global redis_client
    # cdg: redis_client连接方式
    connection_class: type[Union[Connection, SSLConnection]] = Connection
    if dify_config.REDIS_USE_SSL:
        connection_class = SSLConnection
    # cdg: redis_params连接参数
    redis_params: dict[str, Any] = {
        "username": dify_config.REDIS_USERNAME,  # cdg: 用户名
        "password": dify_config.REDIS_PASSWORD,  # cdg: 密码
        "db": dify_config.REDIS_DB,  # cdg: 数据库
        "encoding": "utf-8",
        "encoding_errors": "strict",
        "decode_responses": False,
    }

    # cdg: 使用Sentinel。Sentinel是Redis的哨兵机制，用于监控主节点和从节点，当主节点故障时，Sentinel会自动将一个从节点提升为主节点，从而实现高可用性。
    if dify_config.REDIS_USE_SENTINEL:
        assert dify_config.REDIS_SENTINELS is not None, "REDIS_SENTINELS must be set when REDIS_USE_SENTINEL is True"
        sentinel_hosts = [
            (node.split(":")[0], int(node.split(":")[1])) for node in dify_config.REDIS_SENTINELS.split(",")
        ]
        # cdg: 创建Sentinel实例，设置超时时间、用户名、密码、服务名称
        sentinel = Sentinel(
            sentinel_hosts,
            sentinel_kwargs={
                "socket_timeout": dify_config.REDIS_SENTINEL_SOCKET_TIMEOUT,
                "username": dify_config.REDIS_SENTINEL_USERNAME,
                "password": dify_config.REDIS_SENTINEL_PASSWORD,
            },
        )
        # cdg: 获取主节点
        master = sentinel.master_for(dify_config.REDIS_SENTINEL_SERVICE_NAME, **redis_params)
        # cdg: 初始化Redis客户端
        redis_client.initialize(master)
    elif dify_config.REDIS_USE_CLUSTERS: # cdg: 使用集群，将多个Redis节点组成一个逻辑上的集群，从而实现高可用性和负载均衡。
        assert dify_config.REDIS_CLUSTERS is not None, "REDIS_CLUSTERS must be set when REDIS_USE_CLUSTERS is True"
        nodes = [
            ClusterNode(host=node.split(":")[0], port=int(node.split(":")[1]))
            for node in dify_config.REDIS_CLUSTERS.split(",")
        ]
        # FIXME: mypy error here, try to figure out how to fix it
        redis_client.initialize(RedisCluster(startup_nodes=nodes, password=dify_config.REDIS_CLUSTERS_PASSWORD))  # type: ignore
    else:
        redis_params.update(
            {
                "host": dify_config.REDIS_HOST,
                "port": dify_config.REDIS_PORT,
                "connection_class": connection_class,
            }
        )
        # cdg: 创建连接池，连接池是Redis的连接池，用于管理连接，从而提高性能。
        pool = redis.ConnectionPool(**redis_params)
        # cdg: 初始化Redis客户端
        redis_client.initialize(redis.Redis(connection_pool=pool))
    # cdg: 将Redis客户端添加到Flask应用的扩展中，以便在其他地方使用。
    app.extensions["redis"] = redis_client
