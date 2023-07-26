import pickle
from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Union

import cachetools
import numpy as np
import requests

from gptcache.manager.eviction import EvictionBase
from gptcache.manager.eviction_manager import EvictionManager
from gptcache.manager.object_data.base import ObjectBase
from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    DataType,
    Answer,
    Question,
)
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils.error import CacheError, ParamError
from gptcache.utils.log import gptcache_log


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search
        DataManager管理缓存数据，包括保存和搜索
    """

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs):
        pass

    @abstractmethod
    def import_data(
        self,
        questions: List[Any],
        answers: List[Any],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
    ):
        pass

    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        pass

    def hit_cache_callback(self, res_data, **kwargs):
        pass

    @abstractmethod
    def search(self, embedding_data, **kwargs):
        """search the data in the cache store accrodding to the embedding data
            根据嵌入数据在缓存存储中搜索数据
        抽象方法在父类中被声明但没有具体的实现，需要在子类中进行实现。子类必须提供对抽象方法的具体实现
        :return: a list of search result, [[score, id], [score, id], ...]
        """
        pass

    def flush(self):
        pass

    @abstractmethod
    def add_session(self, res_data, session_id, pre_embedding_data):
        pass

    @abstractmethod
    def list_sessions(self, session_id, key):
        pass

    @abstractmethod
    def delete_session(self, session_id):
        pass

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        pass

    @abstractmethod
    def close(self):
        pass


class MapDataManager(DataManager):
    """MapDataManager, store all data in a map data structure.
        MapDataManager，将所有数据存储在地图数据结构中。
    :param data_path: the path to save the map data, defaults to 'data_map.txt'.
    :type data_path:  str
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable


    Example:
        .. code-block:: python

            from gptcache.manager import get_data_manager

            data_manager = get_data_manager("data_map.txt", 1000)
    """

    def __init__(self, data_path, max_size, get_data_container=None):
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path
        self.init()

    def init(self):
        try:
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            return
        except PermissionError:
            raise CacheError(  # pylint: disable=W0707
                f"You don't have permission to access this file <{self.data_path}>."
            )

    def save(self, question, answer, embedding_data, **kwargs):
        if isinstance(question, Question):
            question = question.content
        session = kwargs.get("session", None)
        session_id = {session.name} if session else set()
        self.data[embedding_data] = (question, answer, embedding_data, session_id)  #直接与文件关联

    def import_data(
        self,
        questions: List[Any],
        answers: List[Any],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        for i, embedding_data in enumerate(embedding_datas):
            self.data[embedding_data] = (
                questions[i],
                answers[i],
                embedding_datas[i],
                {session_ids[i]} if session_ids[i] else set(),
            )

    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        session = kwargs.get("session", None)
        if session:
            answer = (
                res_data[1].answer if isinstance(res_data[1], Answer) else res_data[1]
            )
            if not session.check_hit_func(
                session.name, list(res_data[3]), [res_data[0]], answer
            ):
                return None
        return CacheData(question=res_data[0], answers=res_data[1])

    def search(self, embedding_data, **kwargs):
        try:    #self.data是LRUCache对象;self是MapDataManager对象
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def flush(self):
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        except PermissionError:
            gptcache_log.error(
                "You don't have permission to access this file %s.", self.data_path
            )

    def add_session(self, res_data, session_id, pre_embedding_data):
        res_data[3].add(session_id)

    def list_sessions(self, session_id=None, key=None):
        session_ids = set()
        for k in self.data:
            if session_id and session_id in self.data[k][3]:
                session_ids.add(k)
            elif len(self.data[k][3]) > 0:
                session_ids.update(self.data[k][3])
        return list(session_ids)

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        for k in keys:
            self.data[k][3].remove(session_id)
            if len(self.data[k][3]) == 0:
                del self.data[k]

    def close(self):
        self.flush()


def normalize(vec):
    magnitude = np.linalg.norm(vec)
    normalized_v = vec / magnitude
    return normalized_v


class SSDataManager(DataManager):
    """Generate SSDataManage to manager the data.

    :param s: CacheStorage to manager the scalar data, it can be generated with :meth:`gptcache.manager.CacheBase`.
    #CacheStorage 用于管理标量数据，可以使用 gptcache.manager.CacheBase 生成。
    :type s: CacheStorage
    :param v: VectorBase to manager the vector data, it can be generated with :meth:`gptcache.manager.VectorBase`.
    VectorBase 用于管理矢量数据，可以使用 gptcache.manager.VectorBase 生成。
    :type v:  VectorBase
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param clean_size: the size to clean up, defaults to `max_size * 0.2`.
    :type clean_size: int
    :param eviction: The eviction policy, it is support "LRU" and "FIFO" now, and defaults to "LRU".
    :type eviction:  str
    """

    def __init__(
        self,
        s: CacheStorage,
        v: VectorBase,
        o: Optional[ObjectBase],
        max_size,
        clean_size,
        policy="LRU",
    ):
        self.max_size = max_size
        self.clean_size = clean_size
        self.s = s
        self.v = v
        self.o = o
        self.eviction_base = EvictionBase(
            name="memory",
            policy=policy,
            maxsize=max_size,
            clean_size=clean_size,
            on_evict=self._clear,
        )
        self.eviction_base.put(self.s.get_ids(deleted=False))
        self.eviction_manager = EvictionManager(self.s, self.v) #"驱逐策略"通常在计算机科学中用于描述当内存或缓存空间不足时，系统应如何选择删除哪些数据的策略。

    def _clear(self, marked_keys):
        self.eviction_manager.soft_evict(marked_keys)
        if self.eviction_manager.check_evict(): #判断假删除的个数是不是很多，超过界限就删除
            self.eviction_manager.delete()

    def save(self, question, answer, embedding_data, **kwargs):
        """Save the data and vectors to cache and vector storage.
                将数据和向量保存到缓存和向量存储中。
        :param question: question data.
        :type question: str
        :param answer: answer data.
        :type answer: str, Answer or (Any, DataType)
        :param embedding_data: vector data.
        :type embedding_data: np.ndarray

        Example:
            .. code-block:: python

                import numpy as np
                from gptcache.manager import get_data_manager, CacheBase, VectorBase

                data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
                data_manager.save('hello', 'hi', np.random.random((128, )).astype('float32'))
        """
        session = kwargs.get("session", None)
        session_id = session.name if session else None
        self.import_data([question], [answer], [embedding_data], [session_id])

    def _process_answer_data(self, answers: Union[Answer, List[Answer]]):
        if isinstance(answers, Answer):
            answers = [answers]
        new_ans = []
        for ans in answers:
            if ans.answer_type != DataType.STR:
                new_ans.append(Answer(self.o.put(ans.answer), ans.answer_type))
            else:
                new_ans.append(ans)
        return new_ans

    def _process_question_data(self, question: Union[str, Question]):
        if isinstance(question, Question):
            if question.deps is None:
                return question

            for dep in question.deps:
                if dep.dep_type == DataType.IMAGE_URL:
                    dep.dep_type.data = self.o.put(requests.get(dep.data).content)
            return question

        return Question(question)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Answer],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")

        cache_datas = []
        embedding_datas = [ #list[1]
            normalize(embedding_data) for embedding_data in embedding_datas
        ]
        for i, embedding_data in enumerate(embedding_datas):
            if self.o is not None and not isinstance(answers[i], str):
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]

            cache_datas.append( #本身是新的
                CacheData(
                    question=self._process_question_data(questions[i]),
                    answers=ans,
                    embedding_data=embedding_data.astype("float32"),
                    session_id=session_ids[i],
                )
            )
        ids = self.s.batch_insert(cache_datas)  #皮插入    返回的是插入的数据在数据表中的id值
        print("csdids:")
        print(ids)
        self.v.mul_add(
            [
                VectorData(id=ids[i], data=embedding_data)
                for i, embedding_data in enumerate(embedding_datas)
            ]
        )
        self.eviction_base.put(ids)
#根据给定的数据 ID，获取对应的缓存数据，并进行会话命中检查和答案转换，最后返回获取到的缓存数据对象。
    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        session = kwargs.get("session", None)
        cache_data = self.s.get_data_by_id(res_data[1])
        if cache_data is None:
            return None

        if session:
            cache_answer = (
                cache_data.answers[0].answer
                if isinstance(cache_data.answers[0], Answer)
                else cache_data.answers[0]
            )
            res_list = self.list_sessions(key=res_data[1])
            cache_session_ids, cache_questions = [r.session_id for r in res_list], [
                r.session_question for r in res_list
            ]
            if not session.check_hit_func(
                session.name, cache_session_ids, cache_questions, cache_answer
            ):
                return None

        for ans in cache_data.answers:
            if ans.answer_type != DataType.STR:
                ans.answer = self.o.get(ans.answer)
        return cache_data

    def hit_cache_callback(self, res_data, **kwargs):
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        embedding_data = normalize(embedding_data)
        top_k = kwargs.get("top_k", -1)
        return self.v.search(data=embedding_data, top_k=top_k)

    def flush(self):
        self.s.flush()
        self.v.flush()

    def add_session(self, res_data, session_id, pre_embedding_data):
        self.s.add_session(res_data[1], session_id, pre_embedding_data)

    def list_sessions(self, session_id=None, key=None):
        res = self.s.list_sessions(session_id, key)
        if key:
            return res
        if session_id:
            return list(r.id for r in res)
        return list(set(r.session_id for r in res))

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        self.s.delete_session(keys)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        self.s.report_cache(
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
        )

    def close(self):
        self.s.close()
        self.v.close()
