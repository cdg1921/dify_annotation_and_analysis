import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from configs import dify_config
from core.workflow.graph_engine.entities.run_condition import RunCondition
from core.workflow.nodes import NodeType
from core.workflow.nodes.answer.answer_stream_generate_router import AnswerStreamGeneratorRouter
from core.workflow.nodes.answer.entities import AnswerStreamGenerateRoute
from core.workflow.nodes.end.end_stream_generate_router import EndStreamGeneratorRouter
from core.workflow.nodes.end.entities import EndStreamParam

# cdg:边对象
class GraphEdge(BaseModel):
    source_node_id: str = Field(..., description="source node id")
    target_node_id: str = Field(..., description="target node id")
    run_condition: Optional[RunCondition] = None
    """run condition"""

# cdg:“并行”模块，以对象的方式定义，包括并行模块ID、开始结点、父级并行ID（多层嵌套场景下）、结束结点
class GraphParallel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="random uuid parallel id")
    start_from_node_id: str = Field(..., description="start from node id")
    parent_parallel_id: Optional[str] = None
    """parent parallel id"""
    parent_parallel_start_node_id: Optional[str] = None
    """parent parallel start node id"""
    end_to_node_id: Optional[str] = None
    """end to node id"""

# cdg:DIFY的工作流工作流实际上是一个有向无环图（DAG），因此工作流的结构实际上就是一个图结构，包括根节点、结点列表、边列表、并行关系、结点输出处理关系等
# cdg:DIFY的工作流虽有并行和嵌套模块，但仍可以理解为一个DAG结构
class Graph(BaseModel):
    # cdg:根节点id，"..."在这里的作用是作为一个占位符，表示该字段是必需的，且没有默认值。该字段在实例化时必须提供一个值，否则会引发验证错误。
    root_node_id: str = Field(..., description="root node id of the graph")
    # cdg:结点列表
    node_ids: list[str] = Field(default_factory=list, description="graph node ids")
    # cdg:结点配置列表（字典形式）
    node_id_config_mapping: dict[str, dict] = Field(
        default_factory=list, description="node configs mapping (node id: node config)"
    )
    # cdg:源节点与边的对应关系
    edge_mapping: dict[str, list[GraphEdge]] = Field(
        default_factory=dict, description="graph edge mapping (source node id: edges)"
    )
    # cdg:目标节点与边的对应关系，边的定义详看GraphEdge
    reverse_edge_mapping: dict[str, list[GraphEdge]] = Field(
        default_factory=dict, description="reverse graph edge mapping (target node id: edges)"
    )
    # cdg:并行模块标识，每个并行模块都有一个ID和并行结构
    parallel_mapping: dict[str, GraphParallel] = Field(
        default_factory=dict, description="graph parallel mapping (parallel id: parallel)"
    )
    # cdg:结点与并行结构之间的关系，表示该节点所处的并行模块
    node_parallel_mapping: dict[str, str] = Field(
        default_factory=dict, description="graph node parallel mapping (node id: parallel id)"
    )
    # cdg:两种工作流输出方式：AnswerStreamGenerateRoute和EndStreamParam，
    # end_stream_param在EndStreamProcessor中调用，answer_stream_generate_routes在AnswerStreamProcessor中调用。
    # 调用EndStreamProcessor还是AnswerStreamProcessor详看GraphEngine。GraphEngine在WorkflowEntry中被调用。
    # 综上，WorkflowEntry -> GraphEngine -> [EndStreamProcessor, AnswerStreamProcessor]
    answer_stream_generate_routes: AnswerStreamGenerateRoute = Field(..., description="answer stream generate routes")
    end_stream_param: EndStreamParam = Field(..., description="end stream param")

    # cdg:在Python中，使用双引号（"Graph"）来引用类名是一种常见的做法，尤其是在类型注解中。
    # 这里的双引号用于延迟引用（forward reference）。这意味着在当前上下文中，Graph可能尚未定义，或者在某些情况下，希望在类的定义中引用自身。
    # 使用双引号的好处是，开发人员可以在类的定义中引用该类，而不必担心在类的实际定义之前使用它。这在定义类方法时尤其有用，特别是当类是递归或相互引用的场景
    @classmethod
    def init(cls, graph_config: Mapping[str, Any], root_node_id: Optional[str] = None) -> "Graph":
        """
        Init graph

        :param graph_config: graph config
        :param root_node_id: root node id
        :return: graph
        """
        # cdg:根据配置信息初始化图结构（初始化工作流）

        # edge configs
        edge_configs = graph_config.get("edges")
        if edge_configs is None:
            edge_configs = []

        # cdg:图结构中可以没有边，但不能没有结点，
        # node configs
        node_configs = graph_config.get("nodes")
        if not node_configs:
            raise ValueError("Graph must have at least one node")

        edge_configs = cast(list, edge_configs)
        node_configs = cast(list, node_configs)

        # reorganize edges mapping
        edge_mapping: dict[str, list[GraphEdge]] = {}           # cdg:正向映射关系
        reverse_edge_mapping: dict[str, list[GraphEdge]] = {}   # cdg:反向映射关系
        target_edge_ids = set()
        fail_branch_source_node_id = [
            node["id"] for node in node_configs if node["data"].get("error_strategy") == "fail-branch"
        ]
        # cdg:构建edge_mapping字典
        for edge_config in edge_configs:
            source_node_id = edge_config.get("source")
            # cdg:跳过起始节点为空的边
            if not source_node_id:
                continue

            # cdg:初始化edge_mapping[source_node_id]
            if source_node_id not in edge_mapping:
                edge_mapping[source_node_id] = []

            # cdg:跳过结束节点为空的边
            target_node_id = edge_config.get("target")
            if not target_node_id:
                continue

            # cdg:初始化reverse_edge_mapping[target_node_id]
            if target_node_id not in reverse_edge_mapping:
                reverse_edge_mapping[target_node_id] = []

            target_edge_ids.add(target_node_id)

            # parse run condition
            run_condition = None
            if edge_config.get("sourceHandle"):
                if (
                    edge_config.get("source") in fail_branch_source_node_id
                    and edge_config.get("sourceHandle") != "fail-branch"
                ):
                    run_condition = RunCondition(type="branch_identify", branch_identify="success-branch")
                elif edge_config.get("sourceHandle") != "source":
                    run_condition = RunCondition(
                        type="branch_identify", branch_identify=edge_config.get("sourceHandle")
                    )

            # cdg:实例化GraphEdge
            graph_edge = GraphEdge(
                source_node_id=source_node_id, target_node_id=target_node_id, run_condition=run_condition
            )

            # cdg:赋值给edge_mapping[source_node_id]和reverse_edge_mapping[target_node_id]
            edge_mapping[source_node_id].append(graph_edge)
            reverse_edge_mapping[target_node_id].append(graph_edge)

        # fetch nodes that have no predecessor node
        root_node_configs = []
        all_node_id_config_mapping: dict[str, dict] = {}   # cdg:所有结点配置集合
        for node_config in node_configs:
            node_id = node_config.get("id")
            if not node_id:
                continue

            # cdg:不在结束结点集里，也就是没有边输入，则为根节点
            if node_id not in target_edge_ids:
                root_node_configs.append(node_config)

            all_node_id_config_mapping[node_id] = node_config

        root_node_ids = [node_config.get("id") for node_config in root_node_configs]

        # fetch root node
        if not root_node_id:
            # if no root node id, use the START type node as root node
            # cdg:如果找不到根节点ID，用START类型结点作为根节点
            root_node_id = next(
                (
                    node_config.get("id")
                    for node_config in root_node_configs
                    if node_config.get("data", {}).get("type", "") == NodeType.START.value
                ),
                None,
            )

        if not root_node_id or root_node_id not in root_node_ids:
            raise ValueError(f"Root node id {root_node_id} not found in the graph")

        # cdg:结点间相连关系检查，确保每一个节点都能连接到前一个节点（根节点除外）
        # Check whether it is connected to the previous node
        cls._check_connected_to_previous_node(route=[root_node_id], edge_mapping=edge_mapping)

        # cdg:依次获取所有节点ID
        # fetch all node ids from root node
        node_ids = [root_node_id]
        cls._recursively_add_node_ids(node_ids=node_ids, edge_mapping=edge_mapping, node_id=root_node_id)
        # cdg:由于每个节点都是新增到node_ids列表，_recursively_add_node_ids函数中不需要返回，可以直接调用node_ids

        node_id_config_mapping = {node_id: all_node_id_config_mapping[node_id] for node_id in node_ids}

        # cdg:初始化并发模块字典
        # init parallel mapping
        parallel_mapping: dict[str, GraphParallel] = {}   # cdg:parallel_mapping:{parallel_id, parallel}
        node_parallel_mapping: dict[str, str] = {}        # cdg:node_parallel_mapping:{node_id, parallel_id}
        # cdg:同样，将parallel_mapping、node_parallel_mapping作为参数传给_recursively_add_parallels函数，在函数中更新两个字典，不需要返回
        cls._recursively_add_parallels(
            edge_mapping=edge_mapping,                    # cdg:正向映射，{source_node_id:GraphEdge}
            reverse_edge_mapping=reverse_edge_mapping,    # cdg:反向映射，{target_node_id:GraphEdge}
            start_node_id=root_node_id,
            parallel_mapping=parallel_mapping,
            node_parallel_mapping=node_parallel_mapping,
        )

        # Check if it exceeds N layers of parallel
        for parallel in parallel_mapping.values():
            if parallel.parent_parallel_id:
                # cdg:多级并发检测，默认三级并发
                cls._check_exceed_parallel_limit(
                    parallel_mapping=parallel_mapping,     # parallel_mapping: dict[str, GraphParallel]
                    level_limit=dify_config.WORKFLOW_PARALLEL_DEPTH_LIMIT,
                    parent_parallel_id=parallel.parent_parallel_id,
                )

        # init answer stream generate routes
        answer_stream_generate_routes = AnswerStreamGeneratorRouter.init(
            node_id_config_mapping=node_id_config_mapping, reverse_edge_mapping=reverse_edge_mapping
        )

        # init end stream param
        end_stream_param = EndStreamGeneratorRouter.init(
            node_id_config_mapping=node_id_config_mapping,
            reverse_edge_mapping=reverse_edge_mapping,
            node_parallel_mapping=node_parallel_mapping,
        )

        # 实例化graph对象
        # init graph
        graph = cls(
            root_node_id=root_node_id,
            node_ids=node_ids,
            node_id_config_mapping=node_id_config_mapping,
            edge_mapping=edge_mapping,
            reverse_edge_mapping=reverse_edge_mapping,
            parallel_mapping=parallel_mapping,
            node_parallel_mapping=node_parallel_mapping,
            answer_stream_generate_routes=answer_stream_generate_routes,
            end_stream_param=end_stream_param,
        )

        return graph

    def add_extra_edge(
        self, source_node_id: str, target_node_id: str, run_condition: Optional[RunCondition] = None
    ) -> None:
        """
        Add extra edge to the graph

        :param source_node_id: source node id
        :param target_node_id: target node id
        :param run_condition: run condition
        """
        if source_node_id not in self.node_ids or target_node_id not in self.node_ids:
            return

        if source_node_id not in self.edge_mapping:
            self.edge_mapping[source_node_id] = []

        if target_node_id in [graph_edge.target_node_id for graph_edge in self.edge_mapping[source_node_id]]:
            return

        graph_edge = GraphEdge(
            source_node_id=source_node_id, target_node_id=target_node_id, run_condition=run_condition
        )

        self.edge_mapping[source_node_id].append(graph_edge)

    def get_leaf_node_ids(self) -> list[str]:
        """
        Get leaf node ids of the graph

        :return: leaf node ids
        """
        leaf_node_ids = []
        for node_id in self.node_ids:
            if node_id not in self.edge_mapping or (
                len(self.edge_mapping[node_id]) == 1
                and self.edge_mapping[node_id][0].target_node_id == self.root_node_id
            ):
                leaf_node_ids.append(node_id)

        return leaf_node_ids

    @classmethod
    def _recursively_add_node_ids(
        cls, node_ids: list[str], edge_mapping: dict[str, list[GraphEdge]], node_id: str
    ) -> None:
        """
        Recursively add node ids

        :param node_ids: node ids
        :param edge_mapping: edge mapping
        :param node_id: node id
        """
        for graph_edge in edge_mapping.get(node_id, []):
            if graph_edge.target_node_id in node_ids:
                continue

            # cdg:递归调用_recursively_add_node_ids依次将每条边的target_node_id添加到node_ids列表中；
            node_ids.append(graph_edge.target_node_id)
            cls._recursively_add_node_ids(
                node_ids=node_ids, edge_mapping=edge_mapping, node_id=graph_edge.target_node_id
            )


    @classmethod
    def _check_connected_to_previous_node(cls, route: list[str], edge_mapping: dict[str, list[GraphEdge]]) -> None:
        """
        Check whether it is connected to the previous node
        """
        # cdg:结点链接关系检查，从根节点开始，从前往后检查，例如：A -> B -> C -> D
        # cdg:第一轮：route为[A]，last_node_id=A
        # cdg:第二轮：route为[A, B]，last_node_id=B
        last_node_id = route[-1]

        for graph_edge in edge_mapping.get(last_node_id, []):
            # cdg:第一轮：先找到A，找到graph_edge实例（A, B, run_condition）
            # cdg:第二轮：先找到B，找到graph_edge实例（B, C, run_condition）

            if not graph_edge.target_node_id:
                continue
            # cdg:第一轮：target_node_id为B，不为空
            # cdg:第二轮：target_node_id为C，不为空
            # cdg:……
            # cdg:直到最后一轮，target_node_id为D, graph_edge.target_node_id为空，跳出循环，如果存在环路则会陷入死循环

            # cdg:第一轮：B不在route中，如果出现B在route中，则出现了闭环
            # cdg:第二轮：target_node_id为C，不在route中

            if graph_edge.target_node_id in route:
                raise ValueError(
                    f"Node {graph_edge.source_node_id} is connected to the previous node, please check the graph."
                )

            new_route = route.copy()
            new_route.append(graph_edge.target_node_id)
            # cdg:第一轮：new_route=[A, B]
            # cdg:第二轮：new_route=[A, B, C]
            cls._check_connected_to_previous_node(
                route=new_route,
                edge_mapping=edge_mapping,
            )

    @classmethod
    def _recursively_add_parallels(
        cls,
        edge_mapping: dict[str, list[GraphEdge]],
        reverse_edge_mapping: dict[str, list[GraphEdge]],
        start_node_id: str,
        parallel_mapping: dict[str, GraphParallel],
        node_parallel_mapping: dict[str, str],
        parent_parallel: Optional[GraphParallel] = None,
    ) -> None:
        """
        Recursively add parallel ids

        :param edge_mapping: edge mapping
        :param start_node_id: start from node id
        :param parallel_mapping: parallel mapping
        :param node_parallel_mapping: node parallel mapping
        :param parent_parallel: parent parallel
        """
        # cdg:逐步添加并行模块
        target_node_edges = edge_mapping.get(start_node_id, [])
        parallel = None
        if len(target_node_edges) > 1:
            # fetch all node ids in current parallels
            parallel_branch_node_ids = defaultdict(list)
            condition_edge_mappings = defaultdict(list)
            for graph_edge in target_node_edges:
                # cdg:如果没有指定分支条件，则采用默认值，加到parallel_branch_node_ids中；如果指定，则加到condition_edge_mappings中
                if graph_edge.run_condition is None:
                    parallel_branch_node_ids["default"].append(graph_edge.target_node_id)
                else:
                    condition_hash = graph_edge.run_condition.hash
                    condition_edge_mappings[condition_hash].append(graph_edge)

            for condition_hash, graph_edges in condition_edge_mappings.items():
                if len(graph_edges) > 1:
                    for graph_edge in graph_edges:
                        parallel_branch_node_ids[condition_hash].append(graph_edge.target_node_id)

            condition_parallels = {}
            for condition_hash, condition_parallel_branch_node_ids in parallel_branch_node_ids.items():
                # any target node id in node_parallel_mapping
                parallel = None
                if condition_parallel_branch_node_ids:
                    parent_parallel_id = parent_parallel.id if parent_parallel else None

                    parallel = GraphParallel(
                        start_from_node_id=start_node_id,
                        parent_parallel_id=parent_parallel.id if parent_parallel else None,
                        parent_parallel_start_node_id=parent_parallel.start_from_node_id if parent_parallel else None,
                    )
                    parallel_mapping[parallel.id] = parallel
                    condition_parallels[condition_hash] = parallel

                    in_branch_node_ids = cls._fetch_all_node_ids_in_parallels(
                        edge_mapping=edge_mapping,
                        reverse_edge_mapping=reverse_edge_mapping,
                        parallel_branch_node_ids=condition_parallel_branch_node_ids,
                    )

                    # collect all branches node ids
                    parallel_node_ids = []
                    for _, node_ids in in_branch_node_ids.items():
                        for node_id in node_ids:
                            in_parent_parallel = True
                            if parent_parallel_id:
                                in_parent_parallel = False
                                for parallel_node_id, parallel_id in node_parallel_mapping.items():
                                    if parallel_id == parent_parallel_id and parallel_node_id == node_id:
                                        in_parent_parallel = True
                                        break

                            if in_parent_parallel:
                                parallel_node_ids.append(node_id)
                                node_parallel_mapping[node_id] = parallel.id

                    outside_parallel_target_node_ids = set()
                    for node_id in parallel_node_ids:
                        if node_id == parallel.start_from_node_id:
                            continue

                        node_edges = edge_mapping.get(node_id)
                        if not node_edges:
                            continue

                        if len(node_edges) > 1:
                            continue

                        target_node_id = node_edges[0].target_node_id
                        if target_node_id in parallel_node_ids:
                            continue

                        if parent_parallel_id:
                            parent_parallel = parallel_mapping.get(parent_parallel_id)
                            if not parent_parallel:
                                continue

                        if (
                            (
                                node_parallel_mapping.get(target_node_id)
                                and node_parallel_mapping.get(target_node_id) == parent_parallel_id
                            )
                            or (
                                parent_parallel
                                and parent_parallel.end_to_node_id
                                and target_node_id == parent_parallel.end_to_node_id
                            )
                            or (not node_parallel_mapping.get(target_node_id) and not parent_parallel)
                        ):
                            outside_parallel_target_node_ids.add(target_node_id)

                    if len(outside_parallel_target_node_ids) == 1:
                        if (
                            parent_parallel
                            and parent_parallel.end_to_node_id
                            and parallel.end_to_node_id == parent_parallel.end_to_node_id
                        ):
                            parallel.end_to_node_id = None
                        else:
                            parallel.end_to_node_id = outside_parallel_target_node_ids.pop()

            if condition_edge_mappings:
                for condition_hash, graph_edges in condition_edge_mappings.items():
                    for graph_edge in graph_edges:
                        current_parallel = cls._get_current_parallel(
                            parallel_mapping=parallel_mapping,
                            graph_edge=graph_edge,
                            parallel=condition_parallels.get(condition_hash),
                            parent_parallel=parent_parallel,
                        )

                        cls._recursively_add_parallels(
                            edge_mapping=edge_mapping,
                            reverse_edge_mapping=reverse_edge_mapping,
                            start_node_id=graph_edge.target_node_id,
                            parallel_mapping=parallel_mapping,
                            node_parallel_mapping=node_parallel_mapping,
                            parent_parallel=current_parallel,
                        )
            else:
                for graph_edge in target_node_edges:
                    current_parallel = cls._get_current_parallel(
                        parallel_mapping=parallel_mapping,
                        graph_edge=graph_edge,
                        parallel=parallel,
                        parent_parallel=parent_parallel,
                    )

                    cls._recursively_add_parallels(
                        edge_mapping=edge_mapping,
                        reverse_edge_mapping=reverse_edge_mapping,
                        start_node_id=graph_edge.target_node_id,
                        parallel_mapping=parallel_mapping,
                        node_parallel_mapping=node_parallel_mapping,
                        parent_parallel=current_parallel,
                    )
        else:
            for graph_edge in target_node_edges:
                current_parallel = cls._get_current_parallel(
                    parallel_mapping=parallel_mapping,
                    graph_edge=graph_edge,
                    parallel=parallel,
                    parent_parallel=parent_parallel,
                )

                cls._recursively_add_parallels(
                    edge_mapping=edge_mapping,
                    reverse_edge_mapping=reverse_edge_mapping,
                    start_node_id=graph_edge.target_node_id,
                    parallel_mapping=parallel_mapping,
                    node_parallel_mapping=node_parallel_mapping,
                    parent_parallel=current_parallel,
                )

    @classmethod
    def _get_current_parallel(
        cls,
        parallel_mapping: dict[str, GraphParallel],
        graph_edge: GraphEdge,
        parallel: Optional[GraphParallel] = None,
        parent_parallel: Optional[GraphParallel] = None,
    ) -> Optional[GraphParallel]:
        """
        Get current parallel
        """
        current_parallel = None
        if parallel:
            current_parallel = parallel
        elif parent_parallel:
            if not parent_parallel.end_to_node_id or (
                parent_parallel.end_to_node_id and graph_edge.target_node_id != parent_parallel.end_to_node_id
            ):
                current_parallel = parent_parallel
            else:
                # fetch parent parallel's parent parallel
                parent_parallel_parent_parallel_id = parent_parallel.parent_parallel_id
                if parent_parallel_parent_parallel_id:
                    parent_parallel_parent_parallel = parallel_mapping.get(parent_parallel_parent_parallel_id)
                    if parent_parallel_parent_parallel and (
                        not parent_parallel_parent_parallel.end_to_node_id
                        or (
                            parent_parallel_parent_parallel.end_to_node_id
                            and graph_edge.target_node_id != parent_parallel_parent_parallel.end_to_node_id
                        )
                    ):
                        current_parallel = parent_parallel_parent_parallel

        return current_parallel

    @classmethod
    def _check_exceed_parallel_limit(
        cls,
        parallel_mapping: dict[str, GraphParallel],
        level_limit: int,
        parent_parallel_id: str,
        current_level: int = 1,
    ) -> None:
        """
        Check if it exceeds N layers of parallel
        """
        # cdg:多级并行检测

        parent_parallel = parallel_mapping.get(parent_parallel_id)
        if not parent_parallel:
            return

        current_level += 1
        if current_level > level_limit:
            raise ValueError(f"Exceeds {level_limit} layers of parallel")

        # cdg:如果处于嵌套并行中
        if parent_parallel.parent_parallel_id:
            cls._check_exceed_parallel_limit(
                parallel_mapping=parallel_mapping,
                level_limit=level_limit,
                parent_parallel_id=parent_parallel.parent_parallel_id,
                current_level=current_level,
            )

    @classmethod
    def _recursively_add_parallel_node_ids(
        cls,
        branch_node_ids: list[str],
        edge_mapping: dict[str, list[GraphEdge]],
        merge_node_id: str,
        start_node_id: str,
    ) -> None:
        """
        Recursively add node ids

        :param branch_node_ids: in branch node ids
        :param edge_mapping: edge mapping
        :param merge_node_id: merge node id
        :param start_node_id: start node id
        """
        for graph_edge in edge_mapping.get(start_node_id, []):
            if graph_edge.target_node_id != merge_node_id and graph_edge.target_node_id not in branch_node_ids:
                branch_node_ids.append(graph_edge.target_node_id)
                cls._recursively_add_parallel_node_ids(
                    branch_node_ids=branch_node_ids,
                    edge_mapping=edge_mapping,
                    merge_node_id=merge_node_id,
                    start_node_id=graph_edge.target_node_id,
                )

    @classmethod
    def _fetch_all_node_ids_in_parallels(
        cls,
        edge_mapping: dict[str, list[GraphEdge]],
        reverse_edge_mapping: dict[str, list[GraphEdge]],
        parallel_branch_node_ids: list[str],
    ) -> dict[str, list[str]]:
        """
        Fetch all node ids in parallels
        """
        routes_node_ids: dict[str, list[str]] = {}
        for parallel_branch_node_id in parallel_branch_node_ids:
            routes_node_ids[parallel_branch_node_id] = [parallel_branch_node_id]

            # fetch routes node ids
            cls._recursively_fetch_routes(
                edge_mapping=edge_mapping,
                start_node_id=parallel_branch_node_id,
                routes_node_ids=routes_node_ids[parallel_branch_node_id],
            )

        # fetch leaf node ids from routes node ids
        leaf_node_ids: dict[str, list[str]] = {}
        merge_branch_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            for node_id in node_ids:
                if node_id not in edge_mapping or len(edge_mapping[node_id]) == 0:
                    if branch_node_id not in leaf_node_ids:
                        leaf_node_ids[branch_node_id] = []

                    leaf_node_ids[branch_node_id].append(node_id)

                for branch_node_id2, inner_route2 in routes_node_ids.items():
                    if (
                        branch_node_id != branch_node_id2
                        and node_id in inner_route2
                        and len(reverse_edge_mapping.get(node_id, [])) > 1
                        and cls._is_node_in_routes(
                            reverse_edge_mapping=reverse_edge_mapping,
                            start_node_id=node_id,
                            routes_node_ids=routes_node_ids,
                        )
                    ):
                        if node_id not in merge_branch_node_ids:
                            merge_branch_node_ids[node_id] = []

                        if branch_node_id2 not in merge_branch_node_ids[node_id]:
                            merge_branch_node_ids[node_id].append(branch_node_id2)

        # sorted merge_branch_node_ids by branch_node_ids length desc
        merge_branch_node_ids = dict(sorted(merge_branch_node_ids.items(), key=lambda x: len(x[1]), reverse=True))

        duplicate_end_node_ids = {}
        for node_id, branch_node_ids in merge_branch_node_ids.items():
            for node_id2, branch_node_ids2 in merge_branch_node_ids.items():
                if node_id != node_id2 and set(branch_node_ids) == set(branch_node_ids2):
                    if (node_id, node_id2) not in duplicate_end_node_ids and (
                        node_id2,
                        node_id,
                    ) not in duplicate_end_node_ids:
                        duplicate_end_node_ids[(node_id, node_id2)] = branch_node_ids

        for (node_id, node_id2), branch_node_ids in duplicate_end_node_ids.items():
            # check which node is after
            if cls._is_node2_after_node1(node1_id=node_id, node2_id=node_id2, edge_mapping=edge_mapping):
                if node_id in merge_branch_node_ids and node_id2 in merge_branch_node_ids:
                    del merge_branch_node_ids[node_id2]
            elif cls._is_node2_after_node1(node1_id=node_id2, node2_id=node_id, edge_mapping=edge_mapping):
                if node_id in merge_branch_node_ids and node_id2 in merge_branch_node_ids:
                    del merge_branch_node_ids[node_id]

        branches_merge_node_ids: dict[str, str] = {}
        for node_id, branch_node_ids in merge_branch_node_ids.items():
            if len(branch_node_ids) <= 1:
                continue

            for branch_node_id in branch_node_ids:
                if branch_node_id in branches_merge_node_ids:
                    continue

                branches_merge_node_ids[branch_node_id] = node_id

        in_branch_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            in_branch_node_ids[branch_node_id] = []
            if branch_node_id not in branches_merge_node_ids:
                # all node ids in current branch is in this thread
                in_branch_node_ids[branch_node_id].append(branch_node_id)
                in_branch_node_ids[branch_node_id].extend(node_ids)
            else:
                merge_node_id = branches_merge_node_ids[branch_node_id]
                if merge_node_id != branch_node_id:
                    in_branch_node_ids[branch_node_id].append(branch_node_id)

                # fetch all node ids from branch_node_id and merge_node_id
                cls._recursively_add_parallel_node_ids(
                    branch_node_ids=in_branch_node_ids[branch_node_id],
                    edge_mapping=edge_mapping,
                    merge_node_id=merge_node_id,
                    start_node_id=branch_node_id,
                )

        return in_branch_node_ids

    @classmethod
    def _recursively_fetch_routes(
        cls, edge_mapping: dict[str, list[GraphEdge]], start_node_id: str, routes_node_ids: list[str]
    ) -> None:
        """
        Recursively fetch route
        """
        if start_node_id not in edge_mapping:
            return

        for graph_edge in edge_mapping[start_node_id]:
            # find next node ids
            if graph_edge.target_node_id not in routes_node_ids:
                routes_node_ids.append(graph_edge.target_node_id)

                cls._recursively_fetch_routes(
                    edge_mapping=edge_mapping, start_node_id=graph_edge.target_node_id, routes_node_ids=routes_node_ids
                )

    @classmethod
    def _is_node_in_routes(
        cls, reverse_edge_mapping: dict[str, list[GraphEdge]], start_node_id: str, routes_node_ids: dict[str, list[str]]
    ) -> bool:
        """
        Recursively check if the node is in the routes
        """
        if start_node_id not in reverse_edge_mapping:
            return False

        all_routes_node_ids = set()
        parallel_start_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            all_routes_node_ids.update(node_ids)

            if branch_node_id in reverse_edge_mapping:
                for graph_edge in reverse_edge_mapping[branch_node_id]:
                    if graph_edge.source_node_id not in parallel_start_node_ids:
                        parallel_start_node_ids[graph_edge.source_node_id] = []

                    parallel_start_node_ids[graph_edge.source_node_id].append(branch_node_id)

        for _, branch_node_ids in parallel_start_node_ids.items():
            if set(branch_node_ids) == set(routes_node_ids.keys()):
                return True

        return False

    @classmethod
    def _is_node2_after_node1(cls, node1_id: str, node2_id: str, edge_mapping: dict[str, list[GraphEdge]]) -> bool:
        """
        is node2 after node1
        """
        if node1_id not in edge_mapping:
            return False

        for graph_edge in edge_mapping[node1_id]:
            if graph_edge.target_node_id == node2_id:
                return True

            if cls._is_node2_after_node1(
                node1_id=graph_edge.target_node_id, node2_id=node2_id, edge_mapping=edge_mapping
            ):
                return True

        return False
