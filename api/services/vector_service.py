from typing import Optional

from core.model_manager import ModelInstance, ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.index_type import IndexType
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import Document
from extensions.ext_database import db
from models.dataset import ChildChunk, Dataset, DatasetProcessRule, DocumentSegment
from models.dataset import Document as DatasetDocument
from services.entities.knowledge_entities.knowledge_entities import ParentMode

# cdg: 向量服务，实现向量的创建、更新、删除和子块的生成。RAG模块关键类之一。
class VectorService:
    # cdg: 创建段落向量。这里的段落（segment）指的是文档中的一个文本切块，而不是我们通常理解的自然段落。
    @classmethod
    def create_segments_vector(
        cls, keywords_list: Optional[list[list[str]]], segments: list[DocumentSegment], dataset: Dataset, doc_form: str
    ):
        documents = []

        for segment in segments:
            # cdg: 如果文档格式为父子索引，则需要处理父子关系。
            if doc_form == IndexType.PARENT_CHILD_INDEX:
                document = DatasetDocument.query.filter_by(id=segment.document_id).first()
                # get the process rule
                # cdg: 获取处理规则，如果处理规则不存在，则抛出异常。
                processing_rule = (
                    db.session.query(DatasetProcessRule)
                    .filter(DatasetProcessRule.id == document.dataset_process_rule_id)
                    .first()
                )
                if not processing_rule:
                    raise ValueError("No processing rule found.")
                # get embedding model instance
                # cdg: 针对高质量索引，获取嵌入模型实例，如果嵌入模型实例不存在，则抛出异常。
                if dataset.indexing_technique == "high_quality":
                    # check embedding model setting
                    # cdg: 实例化模型管理器。
                    model_manager = ModelManager()
                    # cdg: 如果嵌入模型提供者存在，则获取嵌入模型实例，否则获取默认嵌入模型实例。
                    if dataset.embedding_model_provider:
                        embedding_model_instance = model_manager.get_model_instance(
                            tenant_id=dataset.tenant_id,
                            provider=dataset.embedding_model_provider,
                            model_type=ModelType.TEXT_EMBEDDING,
                            model=dataset.embedding_model,
                        )
                    else:
                        embedding_model_instance = model_manager.get_default_model_instance(
                            tenant_id=dataset.tenant_id,
                            model_type=ModelType.TEXT_EMBEDDING,
                        )
                else:
                    raise ValueError("The knowledge base index technique is not high quality!")
                # cdg: 调用generate_child_chunks方法，生成子块。
                cls.generate_child_chunks(segment, document, dataset, embedding_model_instance, processing_rule, False)
            else:
                # cdg: 如果文档格式为普通索引（非父子索引），则直接创建文本块。这里的Document对象是一个文本块及其元数据，而非一个文档。
                document = Document(
                    page_content=segment.content,
                    metadata={
                        "doc_id": segment.index_node_id,
                        "doc_hash": segment.index_node_hash,
                        "document_id": segment.document_id,
                        "dataset_id": segment.dataset_id,
                    },
                )
                documents.append(document)
        if len(documents) > 0:
            # cdg: 初始化索引处理器。IndexProcessorFactory中包含ParagraphIndexProcessor、QAIndexProcessor、ParentChildIndexProcessor等3种不同索引方式。
            index_processor = IndexProcessorFactory(doc_form).init_index_processor()
            # cdg: load函数实现加载文档到索引处理器，具体实现方式详看每种索引方式（如ParagraphIndexProcessor）的实现。如/api/core/rag/index_processor/processor/paragraph_index_processor.py中的load方法。
            # cdg: 然后再在load方法中调用向量库vector.create(documents)，将文档添加到向量库中。
            # cdg: create方法的具体实现详看各向量库供应商的实现，默认为weaviate向量库，其路径：/api/core/rag/index_processor/processor/paragraph_index_processor.py
            index_processor.load(dataset, documents, with_keywords=True, keywords_list=keywords_list)

    @classmethod
    def update_segment_vector(cls, keywords: Optional[list[str]], segment: DocumentSegment, dataset: Dataset):
        # update segment index task

        # format new index
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            },
        )
        if dataset.indexing_technique == "high_quality":
            # update vector index
            vector = Vector(dataset=dataset)
            vector.delete_by_ids([segment.index_node_id])
            vector.add_texts([document], duplicate_check=True)

        # update keyword index
        keyword = Keyword(dataset)
        keyword.delete_by_ids([segment.index_node_id])

        # save keyword index
        if keywords and len(keywords) > 0:
            keyword.add_texts([document], keywords_list=[keywords])
        else:
            keyword.add_texts([document])

    @classmethod
    def generate_child_chunks(
        cls,
        segment: DocumentSegment,
        dataset_document: DatasetDocument,
        dataset: Dataset,
        embedding_model_instance: ModelInstance,
        processing_rule: DatasetProcessRule,
        regenerate: bool = False,
    ):
        index_processor = IndexProcessorFactory(dataset.doc_form).init_index_processor()
        if regenerate:
            # delete child chunks
            index_processor.clean(dataset, [segment.index_node_id], with_keywords=True, delete_child_chunks=True)

        # generate child chunks
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            },
        )
        # use full doc mode to generate segment's child chunk
        processing_rule_dict = processing_rule.to_dict()
        processing_rule_dict["rules"]["parent_mode"] = ParentMode.FULL_DOC.value
        documents = index_processor.transform(
            [document],
            embedding_model_instance=embedding_model_instance,
            process_rule=processing_rule_dict,
            tenant_id=dataset.tenant_id,
            doc_language=dataset_document.doc_language,
        )
        # save child chunks
        if documents and documents[0].children:
            index_processor.load(dataset, documents)

            for position, child_chunk in enumerate(documents[0].children, start=1):
                child_segment = ChildChunk(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=dataset_document.id,
                    segment_id=segment.id,
                    position=position,
                    index_node_id=child_chunk.metadata["doc_id"],
                    index_node_hash=child_chunk.metadata["doc_hash"],
                    content=child_chunk.page_content,
                    word_count=len(child_chunk.page_content),
                    type="automatic",
                    created_by=dataset_document.created_by,
                )
                db.session.add(child_segment)
        db.session.commit()

    @classmethod
    def create_child_chunk_vector(cls, child_segment: ChildChunk, dataset: Dataset):
        child_document = Document(
            page_content=child_segment.content,
            metadata={
                "doc_id": child_segment.index_node_id,
                "doc_hash": child_segment.index_node_hash,
                "document_id": child_segment.document_id,
                "dataset_id": child_segment.dataset_id,
            },
        )
        if dataset.indexing_technique == "high_quality":
            # save vector index
            vector = Vector(dataset=dataset)
            vector.add_texts([child_document], duplicate_check=True)

    @classmethod
    def update_child_chunk_vector(
        cls,
        new_child_chunks: list[ChildChunk],
        update_child_chunks: list[ChildChunk],
        delete_child_chunks: list[ChildChunk],
        dataset: Dataset,
    ):
        documents = []
        delete_node_ids = []
        for new_child_chunk in new_child_chunks:
            new_child_document = Document(
                page_content=new_child_chunk.content,
                metadata={
                    "doc_id": new_child_chunk.index_node_id,
                    "doc_hash": new_child_chunk.index_node_hash,
                    "document_id": new_child_chunk.document_id,
                    "dataset_id": new_child_chunk.dataset_id,
                },
            )
            documents.append(new_child_document)
        for update_child_chunk in update_child_chunks:
            child_document = Document(
                page_content=update_child_chunk.content,
                metadata={
                    "doc_id": update_child_chunk.index_node_id,
                    "doc_hash": update_child_chunk.index_node_hash,
                    "document_id": update_child_chunk.document_id,
                    "dataset_id": update_child_chunk.dataset_id,
                },
            )
            documents.append(child_document)
            delete_node_ids.append(update_child_chunk.index_node_id)
        for delete_child_chunk in delete_child_chunks:
            delete_node_ids.append(delete_child_chunk.index_node_id)
        if dataset.indexing_technique == "high_quality":
            # update vector index
            vector = Vector(dataset=dataset)
            if delete_node_ids:
                vector.delete_by_ids(delete_node_ids)
            if documents:
                vector.add_texts(documents, duplicate_check=True)

    @classmethod
    def delete_child_chunk_vector(cls, child_chunk: ChildChunk, dataset: Dataset):
        vector = Vector(dataset=dataset)
        vector.delete_by_ids([child_chunk.index_node_id])
