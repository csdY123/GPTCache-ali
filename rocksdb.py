from rocksdict import Rdict


class rocksdbCacheStorage():
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

    def getColumn_familyConnect(self, column_family):
        """
        :param column_family:  列族名
        :return:   列族对应的连接
        """

        try:
            res = self.db.get_column_family(column_family)
        except(Exception):
            res=self.db.create_column_family(column_family)
        return res
db = rocksdbCacheStorage()
for k, v in db._ques.items():
    print(f"{k} -> {v}----{v.timeLeft}-----{v.create_on}----{v.last_access}-----{v.deleted}")
print("--------------------------------")
for k, v in db._answer.items():
    print(f"{k} -> {v}")
print("--------------------------------")
for k, v in db._ques_dep.items():
    print(f"{k} -> {v}")
print("--------------------------------")
for k, v in db._session.items():
    print(f"{k} -> {v}")
print("--------------------------------")
for k, v in db._report.items():
    print(f"{k} -> {v}")