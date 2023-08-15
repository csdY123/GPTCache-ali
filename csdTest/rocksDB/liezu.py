from rocksdict import Rdict


# 定义列族
column_families = {
    'cf1':str,  # 第一个列族，名称为 "cf1"，使用默认选项
    'cf2':str,  # 第二个列族，名称为 "cf2"，使用默认选项
}

# 打开 RocksDB 数据库，并指定列族
db = Rdict("mydatabase.db", column_families=column_families)

# 通过列族名字获取相应的列族对象
cf1 = db.get_column_family(b'cf1')
cf2 = db.get_column_family(b'cf2')