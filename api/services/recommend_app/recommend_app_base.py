from abc import ABC, abstractmethod

# cdg: 定义RecommendAppRetrievalBase类，用于表示推荐应用检索基类。
class RecommendAppRetrievalBase(ABC):
    """Interface for recommend app retrieval."""

    # cdg: 定义get_recommended_apps_and_categories抽象方法，用于获取推荐应用和类别。
    @abstractmethod
    def get_recommended_apps_and_categories(self, language: str) -> dict:
        raise NotImplementedError

    # cdg: 定义get_recommend_app_detail抽象方法，用于获取推荐应用详情。
    @abstractmethod
    def get_recommend_app_detail(self, app_id: str):
        raise NotImplementedError

    # cdg: 定义get_type抽象方法，用于获取推荐应用类型。
    @abstractmethod
    def get_type(self) -> str:
        raise NotImplementedError
