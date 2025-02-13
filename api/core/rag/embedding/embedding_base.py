from abc import ABC, abstractmethod

# cdg:Embedding的基础类
class Embeddings(ABC):
    """Interface for embedding models."""

    # cdg:对文本段进行embedding
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        raise NotImplementedError

    # cdg:对query进行embedding
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        raise NotImplementedError

    # cdg:异步embedding
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    # cdg:异步query的embedding
    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
