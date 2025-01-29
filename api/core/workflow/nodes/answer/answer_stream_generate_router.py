from core.prompt.utils.prompt_template_parser import PromptTemplateParser
from core.workflow.nodes.answer.entities import (
    AnswerNodeData,
    AnswerStreamGenerateRoute,
    GenerateRouteChunk,
    TextGenerateRouteChunk,
    VarGenerateRouteChunk,
)
from core.workflow.nodes.enums import ErrorStrategy, NodeType
from core.workflow.utils.variable_template_parser import VariableTemplateParser


class AnswerStreamGeneratorRouter:
    @classmethod
    def init(
        cls,
        node_id_config_mapping: dict[str, dict],
        reverse_edge_mapping: dict[str, list["GraphEdge"]],  # type: ignore[name-defined]
    ) -> AnswerStreamGenerateRoute:
        # cdg:返回对象AnswerStreamGenerateRoute与类名AnswerStreamGeneratorRouter之间相差最后一个字符，一个是对象，一个是生成器
        """
        Get stream generate routes.
        :return:
        """
        # parse stream output node value selectors of answer nodes
        answer_generate_route: dict[str, list[GenerateRouteChunk]] = {}
        for answer_node_id, node_config in node_id_config_mapping.items():
            if node_config.get("data", {}).get("type") != NodeType.ANSWER.value:
                continue

            # cdg:_extract_generate_route_selectors函数针对单个节点而言，answer_generate_route包含了所有节点的路由信息
            # get generate route for stream output
            generate_route = cls._extract_generate_route_selectors(node_config)
            answer_generate_route[answer_node_id] = generate_route

        # cdg:输出答案结点的ID列表
        # fetch answer dependencies
        answer_node_ids = list(answer_generate_route.keys())
        # cdg:依次获取生成答案的节点之间的依赖关系
        answer_dependencies = cls._fetch_answers_dependencies(
            answer_node_ids=answer_node_ids,
            reverse_edge_mapping=reverse_edge_mapping,
            node_id_config_mapping=node_id_config_mapping,
        )

        # cdg:返回AnswerStreamGenerateRoute实例
        #     answer_generate_route:(answer node id -> dependent answer node ids)
        #     answer_dependencies:(answer node id -> generate route chunks)
        return AnswerStreamGenerateRoute(
            answer_generate_route=answer_generate_route, answer_dependencies=answer_dependencies
        )

    @classmethod
    def extract_generate_route_from_node_data(cls, node_data: AnswerNodeData) -> list[GenerateRouteChunk]:
        """
        Extract generate route from node data
        :param node_data: node data object
        :return:
        """
        # cdg:从节点输出数据中抽取变量

        # cdg:实例化输出模板
        variable_template_parser = VariableTemplateParser(template=node_data.answer)
        # cdg:变量抽取，variable_selectors格式如下示例：
        # Output: [VariableSelector(variable='#node_id.query.name#', value_selector=['node_id', 'query', 'name']),
        #          VariableSelector(variable='#node_id.query.age#', value_selector=['node_id', 'query', 'age'])]
        variable_selectors = variable_template_parser.extract_variable_selectors()

        # cdg:将variable_selectors转为字典类型，格式如下示例：
        # {'#node_id.query.name#':['node_id', 'query', 'name'], '#node_id.query.age#':['node_id', 'query', 'age']}
        value_selector_mapping = {
            variable_selector.variable: variable_selector.value_selector for variable_selector in variable_selectors
        }

        # cdg:variable_keys格式如下示例：['#node_id.query.name#', '#node_id.query.age#']
        variable_keys = list(value_selector_mapping.keys())

        # cdg:格式化提示词模板
        # format answer template
        template_parser = PromptTemplateParser(template=node_data.answer, with_variable_tmpl=True)
        # cdg:解析提示词模板中的变量名
        template_variable_keys = template_parser.variable_keys

        # cdg:节点配置中的变量名列表与提示词模板中的变量名列表取交集，确定节点配置中哪些变量可以填充到提示词模板中
        # Take the intersection of variable_keys and template_variable_keys
        variable_keys = list(set(variable_keys) & set(template_variable_keys))

        # cdg:利用节点配置信息的变量更新提示词模板
        template = node_data.answer
        for var in variable_keys:
            # cdg:原始模板中变量名有两个“{”，如“{{tools}}”，以下代码中第一个、三、五个“{”是标识符，第五个是标识变量名称
            template = template.replace(f"{{{{{var}}}}}", f"Ω{{{{{var}}}}}Ω")

        generate_routes: list[GenerateRouteChunk] = []
        for part in template.split("Ω"):   # cdg:"Ω"为上述代码中约定的分隔符
            if part:
                if cls._is_variable(part, variable_keys):
                    var_key = part.replace("Ω", "").replace("{{", "").replace("}}", "")
                    value_selector = value_selector_mapping[var_key]
                    generate_routes.append(VarGenerateRouteChunk(value_selector=value_selector))
                else:
                    generate_routes.append(TextGenerateRouteChunk(text=part))

        return generate_routes

    @classmethod
    def _extract_generate_route_selectors(cls, config: dict) -> list[GenerateRouteChunk]:
        """
        Extract generate route selectors
        :param config: node config
        :return:
        """
        # cdg:从配置字典中提取数据并解包为关键字参数，创建一个AnswerNodeData的实例
        node_data = AnswerNodeData(**config.get("data", {}))
        return cls.extract_generate_route_from_node_data(node_data)

    @classmethod
    def _is_variable(cls, part, variable_keys):
        # cdg:判别part是否为变量
        cleaned_part = part.replace("{{", "").replace("}}", "")
        return part.startswith("{{") and cleaned_part in variable_keys

    @classmethod
    def _fetch_answers_dependencies(
        cls,
        answer_node_ids: list[str],
        reverse_edge_mapping: dict[str, list["GraphEdge"]],  # type: ignore[name-defined]
        node_id_config_mapping: dict[str, dict],
    ) -> dict[str, list[str]]:
        """
        Fetch answer dependencies
        :param answer_node_ids: answer node ids
        :param reverse_edge_mapping: reverse edge mapping
        :param node_id_config_mapping: node id config mapping
        :return:
        """
        answer_dependencies: dict[str, list[str]] = {}
        for answer_node_id in answer_node_ids:
            if answer_dependencies.get(answer_node_id) is None:
                answer_dependencies[answer_node_id] = []

            cls._recursive_fetch_answer_dependencies(
                current_node_id=answer_node_id,
                answer_node_id=answer_node_id,
                node_id_config_mapping=node_id_config_mapping,
                reverse_edge_mapping=reverse_edge_mapping,
                answer_dependencies=answer_dependencies,
            )

        return answer_dependencies

    @classmethod
    def _recursive_fetch_answer_dependencies(
        cls,
        current_node_id: str,
        answer_node_id: str,
        node_id_config_mapping: dict[str, dict],
        reverse_edge_mapping: dict[str, list["GraphEdge"]],  # type: ignore[name-defined]
        answer_dependencies: dict[str, list[str]],
    ) -> None:
        """
        Recursive fetch answer dependencies
        :param current_node_id: current node id
        :param answer_node_id: answer node id
        :param node_id_config_mapping: node id config mapping
        :param reverse_edge_mapping: reverse edge mapping
        :param answer_dependencies: answer dependencies
        :return:
        """
        reverse_edges = reverse_edge_mapping.get(current_node_id, [])
        for edge in reverse_edges:
            source_node_id = edge.source_node_id
            if source_node_id not in node_id_config_mapping:
                continue
            source_node_type = node_id_config_mapping[source_node_id].get("data", {}).get("type")
            source_node_data = node_id_config_mapping[source_node_id].get("data", {})
            if (
                source_node_type
                in {
                    NodeType.ANSWER,
                    NodeType.IF_ELSE,
                    NodeType.QUESTION_CLASSIFIER,
                    NodeType.ITERATION,
                    NodeType.VARIABLE_ASSIGNER,
                }
                or source_node_data.get("error_strategy") == ErrorStrategy.FAIL_BRANCH
            ):
                answer_dependencies[answer_node_id].append(source_node_id)
            else:
                cls._recursive_fetch_answer_dependencies(
                    current_node_id=source_node_id,
                    answer_node_id=answer_node_id,
                    node_id_config_mapping=node_id_config_mapping,
                    reverse_edge_mapping=reverse_edge_mapping,
                    answer_dependencies=answer_dependencies,
                )
