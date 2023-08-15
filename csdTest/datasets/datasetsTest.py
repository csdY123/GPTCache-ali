from datasets import Dataset

# 定义CSV文件的路径
csv_file_path = r"E:\GPTCache\问题时间敏感分析\quora-question-pairs\train.csv"

# # 定义列名和数据类型
# column_names = ["id", "qid1", "qid2","question1","question2","is_duplicate"]
# data_types = ["int", "int", "int","string","string","int"]

# 加载CSV文件数据集
dataset = Dataset.from_csv(csv_file_path)

# 查看加载的数据集信息
print(dataset)