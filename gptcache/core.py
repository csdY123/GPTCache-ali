import atexit
import os
from typing import Optional, List, Any

from gptcache.config import Config
from gptcache.embedding.string import to_embeddings as string_embedding
from gptcache.manager import get_data_manager
from gptcache.manager.data_manager import DataManager
from gptcache.processor.post import temperature_softmax
from gptcache.processor.pre import last_content
from gptcache.report import Report
from gptcache.similarity_evaluation import ExactMatchEvaluation
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils import import_openai
from gptcache.utils.cache_func import cache_all
from gptcache.utils.log import gptcache_log


class Cache:
    """GPTCache core object.GPTCache 核心对象。


    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import openai

            cache.init()
            cache.set_openai_key()
    """

    # it should be called when start the cache system
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager: Optional[DataManager] = None
        self.similarity_evaluation: Optional[SimilarityEvaluation] = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report()
        self.next_cache = None

    def init(
        self,
        cache_enable_func=cache_all,
        pre_embedding_func=last_content,
        pre_func=None,
        embedding_func=string_embedding,
        data_manager: DataManager = get_data_manager(),     #调用了子类的get_data_manager()方法获得了一个子类的对象。
        similarity_evaluation=ExactMatchEvaluation(),
        post_process_messages_func=temperature_softmax,
        post_func=None,
        config=Config(),
        next_cache=None,
    ):
        """Pass parameters to initialize GPTCache.

        :param cache_enable_func: a function to enable cache, defaults to ``cache_all`` 启用缓存的函数，默认为``cache_all``
        :param pre_embedding_func: a function to preprocess embedding, defaults to ``last_content``
        :param pre_func: a function to preprocess embedding, same as ``pre_embedding_func``预处理嵌入的函数，与“pre_embedding_func”相同
        :param embedding_func: a function to extract embeddings from requests for similarity search, defaults to ``string_embedding``
        :param data_manager: a ``DataManager`` module, defaults to ``get_data_manager()``
        :param similarity_evaluation: a module to calculate embedding similarity, defaults to ``ExactMatchEvaluation()``
        :param post_process_messages_func: a function to post-process messages, defaults to ``temperature_softmax`` with a default temperature of 0.0
        :param post_func: a function to post-process messages, same as ``post_process_messages_func``
        :param config: a module to pass configurations, defaults to ``Config()``    传递配置的模块，默认为``Config()``
        :param next_cache: customized method for next cache
        """
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_func if pre_func else pre_embedding_func  #首先，它检查 pre_func 是否为真（非零、非空、非 False 等）。如果 pre_func 为真，则将 pre_func 赋值给 self.pre_embedding_func；如果 pre_func 为假，则将 pre_embedding_func 赋值给 self.pre_embedding_func。
        self.embedding_func = embedding_func
        #为了将文本数据转换为适合机器学习、自然语言处理等任务的向量表示形式。预处理函数可以在嵌入之前对文本进行清理和转换，而嵌入函数则负责实际的嵌入过程。
        self.data_manager: DataManager = data_manager
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_func if post_func else post_process_messages_func
        self.config = config
        self.next_cache = next_cache    #下一个缓存的自定义方法

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:  # pylint: disable=W0703
                if not os.getenv("IS_CI"):
                    gptcache_log.error(e)

    def import_data(self, questions: List[Any], answers: List[Any], session_ids: Optional[List[Optional[str]]] = None) -> None:
        """Import data to GPTCache 将数据导入到 GPTCache

        :param questions: preprocessed question Data
        :param answers: list of answers to questions
        :param session_ids: list of the session id.
        :return: None
        """

        self.data_manager.import_data(
            questions=questions,
            answers=answers,
            embedding_datas=[self.embedding_func(question) for question in questions],
            session_ids=session_ids if session_ids else [None for _ in range(len(questions))],
        )

    def flush(self):
        """Flush data, to prevent accidental loss of memory data,
        such as using map cache management or faiss, hnswlib vector storage will be useful
            刷新数据，防止内存数据意外丢失，
            比如使用地图缓存管理或者faiss，hnswlib向量存储将会很有用。
        """
        self.data_manager.flush()
        if self.next_cache:
            self.next_cache.data_manager.flush()

    @staticmethod
    def set_openai_key():
        import_openai()
        import openai  # pylint: disable=C0415

        openai.api_key = os.getenv("OPENAI_API_KEY")


cache = Cache()
