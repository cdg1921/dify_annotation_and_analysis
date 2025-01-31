import threading
from typing import Optional

from flask import Flask, current_app

from core.rag.data_post_processor.data_post_processor import DataPostProcessor
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.embedding.retrieval import RetrievalSegments
from core.rag.index_processor.constant.index_type import IndexType
from core.rag.models.document import Document
from core.rag.rerank.rerank_type import RerankMode
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from extensions.ext_database import db
from models.dataset import ChildChunk, Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument
from services.external_knowledge_service import ExternalDatasetService

default_retrieval_model = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}

# cdg:知识召回关键类
class RetrievalService:
    @classmethod
    def retrieve(
        cls,
        retrieval_method: str,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = 0.0,
        reranking_model: Optional[dict] = None,
        reranking_mode: str = "reranking_model",
        weights: Optional[dict] = None,
    ):
        if not query:
            return []
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return []

        if not dataset or dataset.available_document_count == 0 or dataset.available_segment_count == 0:
            return []
        all_documents: list[Document] = []
        threads: list[threading.Thread] = []
        exceptions: list[str] = []
        # retrieval_model source with keyword
        if retrieval_method == "keyword_search":  # cdg:经济索引模式，关键词倒排索引
            # cdg:创建关键词检索线程
            keyword_thread = threading.Thread(
                target=RetrievalService.keyword_search,
                kwargs={
                    "flask_app": current_app._get_current_object(),  # type: ignore
                    "dataset_id": dataset_id,
                    "query": query,
                    "top_k": top_k,
                    "all_documents": all_documents,                  # cdg:将all_documents作为参数传给检索函数，将召回结果更新到all_documents中，不需要再返回函数执行结果
                    "exceptions": exceptions,
                },
            )
            threads.append(keyword_thread)
            # cdg:启动线程
            keyword_thread.start()

        # cdg:语义索引模式
        # retrieval_model source with semantic
        if RetrievalMethod.is_support_semantic_search(retrieval_method):
            # cdg:创建向量索引线程
            embedding_thread = threading.Thread(
                target=RetrievalService.embedding_search,
                kwargs={
                    "flask_app": current_app._get_current_object(),  # type: ignore
                    "dataset_id": dataset_id,
                    "query": query,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "reranking_model": reranking_model,
                    "all_documents": all_documents,
                    "retrieval_method": retrieval_method,
                    "exceptions": exceptions,
                },
            )
            threads.append(embedding_thread)
            # cdg:启动线程
            embedding_thread.start()

        # cdg:全文检索
        # retrieval source with full text
        if RetrievalMethod.is_support_fulltext_search(retrieval_method):
            # cdg:创建全文检索线程
            full_text_index_thread = threading.Thread(
                target=RetrievalService.full_text_index_search,
                kwargs={
                    "flask_app": current_app._get_current_object(),  # type: ignore
                    "dataset_id": dataset_id,
                    "query": query,
                    "retrieval_method": retrieval_method,
                    "score_threshold": score_threshold,
                    "top_k": top_k,
                    "reranking_model": reranking_model,
                    "all_documents": all_documents,
                    "exceptions": exceptions,
                },
            )
            threads.append(full_text_index_thread)
            # cdg:启动线程
            full_text_index_thread.start()

        # cdg:等待所有线程完成
        for thread in threads:
            thread.join()

        # cdg:将exceptions作为参数传给检索函数，将召回过程若失败，则将失败信息写入exceptions中，不需要再返回函数执行结果
        if exceptions:
            exception_message = ";\n".join(exceptions)
            # cdg:若召回失败，则报错
            raise ValueError(exception_message)

        # cdg:若是混合检索模式（向量检索+全文检索），则需要进行后处理（即重排）
        if retrieval_method == RetrievalMethod.HYBRID_SEARCH.value:
            # cdg:创建reranker实例
            data_post_processor = DataPostProcessor(
                str(dataset.tenant_id), reranking_mode, reranking_model, weights, False
            )
            # cdg:对all_documents的内容进行重排
            all_documents = data_post_processor.invoke(
                query=query,
                documents=all_documents,
                score_threshold=score_threshold,
                top_n=top_k,
            )

        return all_documents

    # cdg:从外部知识库进行召回
    @classmethod
    def external_retrieve(cls, dataset_id: str, query: str, external_retrieval_model: Optional[dict] = None):
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return []
        all_documents = ExternalDatasetService.fetch_external_knowledge_retrieval(
            dataset.tenant_id, dataset_id, query, external_retrieval_model or {}
        )
        return all_documents

    # cdg:关键词检索（倒排索引）
    @classmethod
    def keyword_search(
        cls, flask_app: Flask, dataset_id: str, query: str, top_k: int, all_documents: list, exceptions: list
    ):
        with flask_app.app_context():
            try:
                dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
                if not dataset:
                    raise ValueError("dataset not found")

                # cdg:创建关键词实例
                keyword = Keyword(dataset=dataset)

                # cdg:执行关键词检索
                documents = keyword.search(cls.escape_query_for_search(query), top_k=top_k)
                # cdg:将关键词检索结果添加到all_documents中，all_documents作为参数传入，在函数中更新all_documents，不需要返回结果了
                all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    # cdg:向量检索（语义检索），在向量检索或者混合检索模式下，都会执行本函数
    @classmethod
    def embedding_search(
        cls,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        reranking_model: Optional[dict],
        all_documents: list,
        retrieval_method: str,
        exceptions: list,
    ):
        with flask_app.app_context():
            try:
                dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
                if not dataset:
                    raise ValueError("dataset not found")

                # 创建向量检索实例
                vector = Vector(dataset=dataset)

                # 执行向量召回
                documents = vector.search_by_vector(
                    cls.escape_query_for_search(query),
                    search_type="similarity_score_threshold",
                    top_k=top_k,
                    score_threshold=score_threshold,
                    filter={"group_id": [dataset.id]},
                )

                # cdg:如果是语义检索，召回结果不为空，而且要求执行重排，则进行重排；在混合检索模式中，不会执行以下代码，只有在单纯的向量检索模式中才会执行以下代码
                if documents:
                    if (
                        reranking_model
                        and reranking_model.get("reranking_model_name")
                        and reranking_model.get("reranking_provider_name")
                        and retrieval_method == RetrievalMethod.SEMANTIC_SEARCH.value
                    ):
                        # cdg:创建后处理器（Reranker处理器）
                        data_post_processor = DataPostProcessor(
                            str(dataset.tenant_id), RerankMode.RERANKING_MODEL.value, reranking_model, None, False
                        )
                        # cdg:将重排后的结果加入all_documents中
                        all_documents.extend(
                            data_post_processor.invoke(
                                query=query,
                                documents=documents,
                                score_threshold=score_threshold,
                                top_n=len(documents),
                            )
                        )
                    else:
                        all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    # cdg:全文检索，在全文检索或混合检索模式下，才会执行该函数
    @classmethod
    def full_text_index_search(
        cls,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        reranking_model: Optional[dict],
        all_documents: list,
        retrieval_method: str,
        exceptions: list,
    ):
        with flask_app.app_context():
            try:
                dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
                if not dataset:
                    raise ValueError("dataset not found")

                # cdg:创建向量处理器
                vector_processor = Vector(
                    dataset=dataset,
                )

                # cdg:执行全文检索，全文检索的功能是在Vector对象中实现，不是所有向量库都支持全文检索，目前Weaviate支持全文检索，MIlvus向量库在2.5版本之后才支持该功能
                documents = vector_processor.search_by_full_text(cls.escape_query_for_search(query), top_k=top_k)
                # cdg:单纯全文检索模式才会执行以下代码，混合检索模式不会执行
                if documents:
                    if (
                        reranking_model
                        and reranking_model.get("reranking_model_name")
                        and reranking_model.get("reranking_provider_name")
                        and retrieval_method == RetrievalMethod.FULL_TEXT_SEARCH.value
                    ):
                        data_post_processor = DataPostProcessor(
                            str(dataset.tenant_id), RerankMode.RERANKING_MODEL.value, reranking_model, None, False
                        )
                        all_documents.extend(
                            data_post_processor.invoke(
                                query=query,
                                documents=documents,
                                score_threshold=score_threshold,
                                top_n=len(documents),
                            )
                        )
                    else:
                        all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    # cdg:替换字符串中的双引号，在DIFY较低的一些版本中（如0.6），没有该函数，如果Query中包含双引号会报错
    @staticmethod
    def escape_query_for_search(query: str) -> str:
        return query.replace('"', '\\"')

    @staticmethod
    def format_retrieval_documents(documents: list[Document]) -> list[RetrievalSegments]:
        records = []
        include_segment_ids = []
        segment_child_map = {}
        for document in documents:
            document_id = document.metadata.get("document_id")
            dataset_document = db.session.query(DatasetDocument).filter(DatasetDocument.id == document_id).first()
            if dataset_document:
                if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
                    child_index_node_id = document.metadata.get("doc_id")
                    result = (
                        db.session.query(ChildChunk, DocumentSegment)
                        .join(DocumentSegment, ChildChunk.segment_id == DocumentSegment.id)
                        .filter(
                            ChildChunk.index_node_id == child_index_node_id,
                            DocumentSegment.dataset_id == dataset_document.dataset_id,
                            DocumentSegment.enabled == True,
                            DocumentSegment.status == "completed",
                        )
                        .first()
                    )
                    if result:
                        child_chunk, segment = result
                        if not segment:
                            continue
                        if segment.id not in include_segment_ids:
                            include_segment_ids.append(segment.id)
                            child_chunk_detail = {
                                "id": child_chunk.id,
                                "content": child_chunk.content,
                                "position": child_chunk.position,
                                "score": document.metadata.get("score", 0.0),
                            }
                            map_detail = {
                                "max_score": document.metadata.get("score", 0.0),
                                "child_chunks": [child_chunk_detail],
                            }
                            segment_child_map[segment.id] = map_detail
                            record = {
                                "segment": segment,
                            }
                            records.append(record)
                        else:
                            child_chunk_detail = {
                                "id": child_chunk.id,
                                "content": child_chunk.content,
                                "position": child_chunk.position,
                                "score": document.metadata.get("score", 0.0),
                            }
                            segment_child_map[segment.id]["child_chunks"].append(child_chunk_detail)
                            segment_child_map[segment.id]["max_score"] = max(
                                segment_child_map[segment.id]["max_score"], document.metadata.get("score", 0.0)
                            )
                    else:
                        continue
                else:
                    index_node_id = document.metadata["doc_id"]

                    segment = (
                        db.session.query(DocumentSegment)
                        .filter(
                            DocumentSegment.dataset_id == dataset_document.dataset_id,
                            DocumentSegment.enabled == True,
                            DocumentSegment.status == "completed",
                            DocumentSegment.index_node_id == index_node_id,
                        )
                        .first()
                    )

                    if not segment:
                        continue
                    include_segment_ids.append(segment.id)
                    record = {
                        "segment": segment,
                        "score": document.metadata.get("score", None),
                    }

                    records.append(record)
            for record in records:
                if record["segment"].id in segment_child_map:
                    record["child_chunks"] = segment_child_map[record["segment"].id].get("child_chunks", None)
                    record["score"] = segment_child_map[record["segment"].id]["max_score"]

        return [RetrievalSegments(**record) for record in records]
