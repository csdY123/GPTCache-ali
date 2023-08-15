import rocksdb3


path = 'db_path'    #给出路径

db = rocksdb3.open_default(path)    #打开数据库
if db.get(b'my key'):
    pass
else:
    print(db.get(b'my key'))

db.put(b'my key', '陈森达'.encode())      #放入数据
if db.get(b'my key'):
    print(db.get(b'my key').decode())
else:
    print(db.get(b'my key').decode())

# assert list(db.get_iter()) == [(b'my key', b'my value')]    #在干什么？
db.delete(b'my key')                #删除key为："my key"的数据
assert db.get(b'my key') is None    #得到key为：”my key“的数据
del db  # auto close db             #删除数据库
rocksdb3.destroy(path)              #毁灭路径
