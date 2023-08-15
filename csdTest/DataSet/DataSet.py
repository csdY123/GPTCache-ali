# 重写dataset类 无用
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class myDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: 数据文件路径
        """
        # 读文件夹下每个数据文件的名称
        # self.file_name = os.listdir(data_dir)
        self.file_name=data_dir
        # 把每一个文件的路径拼接起来
        # for index in range(len(self.file_name)):
        #     self.data_path.append(os.path.join(data_dir, self.file_name[index]))

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        # 读取每一个数据
        data = pd.read_csv(self.file_name, header=None)
        # # 转成张量
        # # data = torch.tensor(data.values)
        # question1=data[3]
        # question2 = data[4]
        # isdup = data[5]
        return {'text': data[3][index]+" |分隔| "+data[4][index],'label':data[5][index]}


# 实例化，读取数据

in_dir = r"E:\GPTCache\问题时间敏感分析\quora-question-pairs\train.csv"
# 读取数据集
train_dataset = myDataset(data_dir=in_dir)
for data in train_dataset:
    print(data)
# 加载数据集
# print(train_dataset.map())
# tokenized_imdb = train_dataset.map(batched=True)
# train_iter = DataLoader(train_dataset[0][2], batch_size=4)
# print(train_dataset[0][2])
# for i in train_iter:
#     print(i)