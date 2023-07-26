from apscheduler.schedulers.blocking import BlockingScheduler

def my_task():
    print("定时任务执行啦！")

# 创建调度器
scheduler = BlockingScheduler()

# 每隔一分钟执行一次任务
scheduler.add_job(my_task, 'interval', minutes=0.1)

try:
    scheduler.start()
except KeyboardInterrupt:
    pass
