from typing import cast

from core.workflow.nodes import NodeType
from core.workflow.nodes.knowledge_retrieval.entities import KnowledgeRetrievalNodeData
from events.app_event import app_published_workflow_was_updated
from extensions.ext_database import db
from models.dataset import AppDatasetJoin
from models.workflow import Workflow

# cdg: 使用@app_published_workflow_was_updated.connect装饰器，将handle函数注册为应用发布工作流更新事件的处理器。
@app_published_workflow_was_updated.connect
def handle(sender, **kwargs): # cdg: 处理应用发布工作流更新事件，sender是应用对象，kwargs是事件参数 
    app = sender # cdg: 获取应用对象
    published_workflow = kwargs.get("published_workflow") # cdg: 获取发布的工作流
    published_workflow = cast(Workflow, published_workflow) # cdg: 将发布的工作流转换为Workflow对象

    dataset_ids = get_dataset_ids_from_workflow(published_workflow) # cdg: 获取数据集ID列表
    app_dataset_joins = db.session.query(AppDatasetJoin).filter(AppDatasetJoin.app_id == app.id).all() # cdg: 获取应用数据集关联

    removed_dataset_ids: set[str] = set() # cdg: 初始化删除的数据集ID列表
    if not app_dataset_joins: # cdg: 如果应用数据集关联为空，则设置添加的数据集ID列表为数据集ID列表
        added_dataset_ids = dataset_ids
    else: # cdg: 如果应用数据集关联不为空，则设置添加的数据集ID列表为数据集ID列表减去旧的数据集ID列表
        old_dataset_ids: set[str] = set() # cdg: 初始化旧的数据集ID列表
        old_dataset_ids.update(app_dataset_join.dataset_id for app_dataset_join in app_dataset_joins) # cdg: 更新

        added_dataset_ids = dataset_ids - old_dataset_ids # cdg: 设置添加的数据集ID列表为数据集ID列表减去旧的数据集ID列表
        removed_dataset_ids = old_dataset_ids - dataset_ids # cdg: 设置删除的数据集ID列表为旧的数据集ID列表减去数据集ID列表

    if removed_dataset_ids: # cdg: 如果删除的数据集ID列表不为空，则删除应用数据集关联
        for dataset_id in removed_dataset_ids: # cdg: 遍历删除的数据集ID列表
            db.session.query(AppDatasetJoin).filter( # cdg: 删除应用数据集关联
                AppDatasetJoin.app_id == app.id, AppDatasetJoin.dataset_id == dataset_id
            ).delete() # cdg: 删除应用数据集关联

    if added_dataset_ids: # cdg: 如果添加的数据集ID列表不为空，则添加应用数据集关联
        for dataset_id in added_dataset_ids: # cdg: 遍历添加的数据集ID列表
            app_dataset_join = AppDatasetJoin(app_id=app.id, dataset_id=dataset_id) # cdg: 创建应用数据集关联对象
            db.session.add(app_dataset_join) # cdg: 添加应用数据集关联对象到数据库

    db.session.commit() # cdg: 提交应用数据集关联对象到数据库

# cdg: 从工作流中获取数据集ID列表
def get_dataset_ids_from_workflow(published_workflow: Workflow) -> set[str]:
    dataset_ids: set[str] = set()
    graph = published_workflow.graph_dict
    if not graph:
        return dataset_ids

    nodes = graph.get("nodes", [])

    # fetch all knowledge retrieval nodes
    knowledge_retrieval_nodes = [
        node for node in nodes if node.get("data", {}).get("type") == NodeType.KNOWLEDGE_RETRIEVAL.value
    ]

    if not knowledge_retrieval_nodes:
        return dataset_ids

    for node in knowledge_retrieval_nodes:
        try:
            node_data = KnowledgeRetrievalNodeData(**node.get("data", {}))
            dataset_ids.update(dataset_id for dataset_id in node_data.dataset_ids)
        except Exception as e:
            continue

    return dataset_ids
