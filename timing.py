from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding import Onnx
onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
def delete_OutdatedData():      #每隔一分钟删除一次五分钟之内需要删除的数据
    AllIds=data_manager.s.get_ids(deleted=False)
    print(AllIds)
    for i in AllIds:
        a=data_manager.s.get_data_by_id(i)
        print(a.timeLeft)
        if a.timeLeft<5:
            data_manager.s.mark_one_deleted(i)
            ii=list()
            ii.append(i)
            data_manager.v.delete(ii)
            data_manager.v.flush()
            data_manager.s.flush()
def get_minutes_passed(datetime_object):
    # 获取当前时间的datetime.datetime对象
    current_datetime = datetime.now()
    # 计算时间差
    time_difference = current_datetime - datetime_object
    # 提取分钟部分
    minutes_passed = time_difference.total_seconds() // 60
    #print("过去了 {} 分钟".format(minutes_passed))
    return minutes_passed

def delete_AllData():      #每个一个小时删除一次过时的数据。
    data_manager.s.clear_deleted_data()   #删除设置为-1，即假删除的数据
    AllIds=data_manager.s.get_ids(deleted=False)
    print(AllIds)
    for i in AllIds:
        a=data_manager.s.get_data_by_id(i)
        minutes_passed=get_minutes_passed(a.last_access)
        if minutes_passed>=55:
            data_manager.s.mark_one_deleted(i)
            ii=list()
            ii.append(i)
            data_manager.v.delete(ii)
            data_manager.v.flush()
            data_manager.s.flush()

# def my_task():
#     print("定时任务执行啦！")
# # 每隔一分钟执行一次任务
# scheduler.add_job(my_task, 'interval', minutes=0.1)
# delete_AllData()
delete_OutdatedData()
# 创建调度器
scheduler = BlockingScheduler()


# 每隔一分钟删除一次五分钟之内需要删除的数据
scheduler.add_job(delete_OutdatedData, 'interval', minutes=1)
#每个一个小时删除一次过时的数据。
scheduler.add_job(delete_AllData, 'interval', minutes=59.5)

try:
    scheduler.start()
except KeyboardInterrupt:
    pass
