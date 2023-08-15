# 导入数据支持
import datetime
from typing import List

from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    Question,
    QuestionDep,
)
# 导入rocksdb库
from rocksdict import Rdict
import numpy as np
import pandas as pd

from gptcache.utils import import_rocksdb

import_rocksdb()


class Answers():
    def __init__(self, answer: str, answer_type: int):
        self.answer = answer
        self.answer_type = answer_type


class Questions():
    def __init__(self, question, create_on: datetime.datetime,
                 last_access: datetime.datetime, deleted: int, timeLeft: int = 60, answers: List[Answers] = None):
        self.question = question
        self.create_on = create_on
        self.last_access = last_access
        self.deleted = deleted
        self.answers = answers
        self.timeLeft = timeLeft


class Ques_Dep():
    def __init__(self, dep_name: str, dep_data: str, dep_type: int):
        self.dep_name = dep_name
        self.dep_data = dep_data
        self.dep_type = dep_type


class Sessions():
    def __init__(self, session_id: int, session_question: str):
        self.session_id = session_id,
        self.session_question = session_question


class Report:
    def __init__(self, user_question: str, cache_question_id: int, cache_question: str, cache_answer: str,
                 similarity: float, cache_delta_time: float, cache_time: datetime.datetime, extra: str = None):
        self.user_question = user_question
        self.cache_question_id = cache_question_id
        self.cache_question = cache_question
        self.cache_answer = cache_answer
        self.similarity = similarity
        self.cache_delta_time = cache_delta_time
        self.cache_time = cache_time
        self.extra = extra


class rocksdbCacheStorage(CacheStorage):
    def __init__(
            self,
            path: str = "./test_dict",
            table_name: str = "gptcache",
            column_family1: str = "question",
            column_family2: str = "answer",
            column_family3: str = "ques_dep",
            column_family4: str = "session",
            column_family5: str = "report",
            **kwargs
    ):
        self.path = path
        self.table_name = table_name
        self.db = Rdict(path)
        self._ques = self.getColumn_familyConnect(column_family1)
        self._answer = self.getColumn_familyConnect(column_family2)
        self._ques_dep = self.getColumn_familyConnect(column_family3)
        self._session = self.getColumn_familyConnect(column_family4)
        self._report = self.getColumn_familyConnect(column_family5)
        self.nextNum = self.getMaxNum() + 1
        self.nextReportNum = self.getReportNum() + 1

    def getReportNum(self):
        maxNum = -1
        for k, v in self._report.items():
            if maxNum < k:
                maxNum = k
        # print("maxNum:",maxNum)
        return maxNum

    def getColumn_familyConnect(self, column_family):
        """
        :param column_family:  列族名
        :return:   列族对应的连接
        """

        try:
            res = self.db.get_column_family(column_family)
        except(Exception):
            res = self.db.create_column_family(column_family)
        return res

    def getMaxNum(self):
        """
        :return: 返回DB中key最大的那个值
        """
        maxNum = -1
        for k, v in self._ques.items():
            if maxNum < k:
                maxNum = k
        return maxNum

    def create(self):
        """
        没有就得创造
        create在初始化时已经创建，不需要重复创建
        :return:
        """
        pass

    def _insert(self, data: CacheData):
        """
        插入数据
        :param data: 需要插入的数据 是CacheData类型的数据
        :return: 返回key值
        """
        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        for answer in answers:
            self._answer[self.nextNum] = Answers(
                answer=answer.answer,
                answer_type=int(answer.answer_type),
            )

        self._ques[self.nextNum] = Questions(
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            create_on=datetime.datetime.utcnow(),
            last_access=datetime.datetime.utcnow(),
            deleted=0,
            answers=data.embedding_data.astype(np.float32).tobytes()
        )
        # embedding_data事实上，不需要存储embedding_data
        if isinstance(data.question, Question) and data.question.deps is not None:
            for dep in data.question.deps:
                self._ques_dep[self.nextNum] = Ques_Dep(
                    dep_name=dep.name,
                    dep_data=dep.data,
                    dep_type=dep.dep_type,
                )
        if data.session_id:
            self._session[self.nextNum] = Sessions(
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
        self.nextNum = self.nextNum + 1
        return int(self.nextNum - 1)

    def batch_insert(self, all_data: List[CacheData]):
        """
        批量插入
        :param all_data: 批量插入的数据
        :return:
        """
        ids = []
        for data in all_data:
            ids.append(self._insert(data))
        return ids

    def get_data_by_id(self, key: int):
        """
        :param key: key为 key对应的值
        :return: key对应的value
        """
        temp = self._ques[key]
        temp.last_access = datetime.datetime.utcnow()
        temp.timeLeft = temp.timeLeft / 2
        self._ques[key] = temp

        self._ques.flush()
        questioncsd = self._ques[key]
        answer = self._answer[key]
        res_ans = [(answer.answer, answer.answer_type)]
        # ques_dep = self._ques_dep[key]
        if self._session.key_may_exist(key):
            session = self._session[key]
        else:
            session = Sessions(session_id=-1, session_question="null")
        return CacheData(
            question=questioncsd.question,
            answers=res_ans,
            embedding_data=np.frombuffer(questioncsd.answers, dtype=np.float32),
            session_id=session.session_id,
            create_on=questioncsd.create_on,
            last_access=questioncsd.last_access,
        )

    def get_ids(self, deleted=True):
        """
        :param deleted: 是否是得到需要删除的id列表
        :return:
        """
        state = -1 if deleted else 0
        res = []
        for k, v in self._ques.items():
            if v.deleted == state:
                res.append(int(k))
        return res

    def mark_one_deleted(self, key):
        """
        :param key: 为需要删除的key
        :return:
        """
        temp = self._ques[key]
        temp.deleted = -1
        self._ques[key] = temp

    def mark_deleted(self, keys):
        """
        :param keys:  需要删除的key列表
        :return:
        """
        for index in keys:
            temp=self._ques[index]
            temp.deleted = -1
            self._ques[index] = temp
    def clear_deleted_data(self):
        """
        删除被标记为需要删除的记录
        :return:
        """
        for k, v in self._ques.items():
            if v.deleted == -1:
                self._ques.delete(k)

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            count = 0
            for k, v in self._ques.items():
                count = count + 1
            return count
        else:
            count = 0
            for k, v in self._ques.items():
                if v.deleted == state:
                    count = count + 1
            return count

    def add_session(self, question_id, session_id, session_question):
        self._session[question_id] = Sessions(session_id=session_id,
                                              session_question=session_question)

    def delete_session(self, keys):
        keys = [int(key) for key in keys]
        for index in keys:
            self._session.delete(index)

    def list_sessions(self, session_id=None, key=None):
        pass

    def report_cache(self, user_question, cache_question, cache_question_id, cache_answer, similarity_value,
                     cache_delta_time):
        """
        记录cache命中的情况
        :param user_question:
        :param cache_question:
        :param cache_question_id:
        :param cache_answer:
        :param similarity_value:
        :param cache_delta_time:
        :return:
        """
        # print("self.nextReportNum:", self.nextReportNum)
        self._report[self.nextReportNum] = Report(
            user_question=user_question,
            cache_question=cache_question,
            cache_question_id=cache_question_id,
            cache_answer=cache_answer,
            similarity=similarity_value,
            cache_delta_time=cache_delta_time,
            cache_time=datetime.datetime.utcnow(),
        )
        self.nextReportNum = self.nextReportNum + 1

    def close(self):
        """
        关闭rocksdb
        :return:
        """
        self._ques.close()
        self._answer.close()
        self._ques_dep.close()
        self._session.close()
        self._report.close()
