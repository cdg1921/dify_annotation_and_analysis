"""Abstract interface for document loader implementations."""

from core.rag.index_processor.constant.index_type import IndexType
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from core.rag.index_processor.processor.parent_child_index_processor import ParentChildIndexProcessor
from core.rag.index_processor.processor.qa_index_processor import QAIndexProcessor


class IndexProcessorFactory:
    """IndexProcessorInit."""

    def __init__(self, index_type: str | None):
        self._index_type = index_type

    def init_index_processor(self) -> BaseIndexProcessor:
        """Init index processor."""

        if not self._index_type:
            raise ValueError("Index type must be specified.")

        # cdg:不同的文档处理方式得到不同的知识库，具体采用哪种文本处理方式合适，一般看数据情况以及需求情况
        if self._index_type == IndexType.PARAGRAPH_INDEX:         # cdg:文本分段
            return ParagraphIndexProcessor()
        elif self._index_type == IndexType.QA_INDEX:              # cdg:QA对生成
            return QAIndexProcessor()
        elif self._index_type == IndexType.PARENT_CHILD_INDEX:    # cdg:句子滑窗
            return ParentChildIndexProcessor()
        else:
            raise ValueError(f"Index type {self._index_type} is not supported.")
