from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Any


class SimilarityEvaluation(metaclass=ABCMeta):
    """Similarity Evaluation interface,
    determine the similarity between the input request and the requests from the Vector Store.
    Based on this similarity, it determines whether a request matches the cache.
    相似度评估界面，
     确定输入请求和来自矢量存储的请求之间的相似性。
     根据这种相似性，它确定请求是否与缓存匹配。
    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.similarity_evaluation import SearchDistanceEvaluation

            cache.init(
                similarity_evaluation=SearchDistanceEvaluation()
            )
    """

    @abstractmethod
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """Evaluate the similarity score of the user and cache requests pair.

        :param src_dict: the user request params.
        :type src_dict: Dict
        :param cache_dict: the cache request params.
        :type cache_dict: Dict
        """
        pass

    @abstractmethod
    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: the range of similarity score, which is the min and max values
        :rtype: Tuple[float, float]
        """
        pass
