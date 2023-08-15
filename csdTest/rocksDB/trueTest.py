import datetime
from typing import List

from rocksdict import Rdict
import numpy as np
import pandas as pd

path = str("./test_dict")

# create a Rdict with default options at `path`
db = Rdict(path)

# db.create_column_family("cf1")
try:
    db_Cf1 = db.get_column_family("a")
except(Exception):
    db.create_column_family("a")
db_Cf1 = db.get_column_family("a")
print("成果：", db_Cf1)

print(db_Cf1)

db[1.0] = 1
db["huge integer"] = 2343546543243564534233536434567543
db["huge integer"] = 1234444444444444444444444444444444
db["good"] = True
db["bytes"] = b"bytes"
db["this is a list"] = [1, 2, 3]
db["store a dict"] = {0: 1}
db[b"numpy"] = np.array([1, 2, 3])  # 键使用了字节字符串的表示方式
db["a table"] = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
db_Cf1[1] = 1
db_Cf1[2] = 2
db_Cf1[3] = 3

# reopen Rdict from disk
db.close()
db_Cf1.close()
db = Rdict(path)
db_Cf1 = db.get_column_family("cf1")
# assert db[1.0] == 1
# assert db["huge integer"] == 2343546543243564534233536434567543
# assert db["good"] == True
# assert db["bytes"] == b"bytes"
# assert db["this is a list"] == [1, 2, 3]
# assert db["store a dict"] == {0: 1}
# assert np.all(db[b"numpy"] == np.array([1, 2, 3]))
# assert np.all(db["a table"] == pd.DataFrame({"a": [1, 2], "b": [2, 1]}))


class Answers():
    answer: str
    answer_type: int


class Questions():
    def __init__(self, question, create_on: datetime.datetime,
                 last_access: datetime.datetime, deleted: int,timeLeft: int= 60 , answers: List[Answers] = None):
        self.question = question
        self.create_on = create_on
        self.last_access = last_access
        self.deleted = deleted
        self.answers = answers
        self.timeLeft = timeLeft


q = Questions(
    question="questions",
    create_on=datetime.datetime,
    last_access=datetime.datetime,
    deleted=0,
    timeLeft=60,
    answers=[

    ]
)
db[123] = q
db.delete(123)

# iterate through all elements
for k, v in db.items():
    print(f"{k} -> {v}")
    if isinstance(v, Questions):
        print(v.create_on)
print("_______________________________________________________")
for k, v in db_Cf1.items():
    print(f"{k} -> {v}")

# batch get:
print(db[["good", "bad", 1.0]])
print(db.key_may_exist("gocod"))
# [True, False, 1]

# delete Rdict from dict
db.close()
# Rdict.destroy(path)
