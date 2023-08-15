# -*- coding: utf-8 -*-
import csv
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers.util import cos_sim
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh',cache_dir="..\csdTest")
model = AutoModel.from_pretrained('BAAI/bge-large-zh',cache_dir="..\csdTest")

def getQuestions():
    data_list = []
    csv_file_path = r'E:\GPTCache\问题时间敏感分析\quora-question-pairs\train.csv'

    # Open the CSV file in read mode
    with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Skip the header row (first row)
        next(reader)

        # Iterate over each row and store the values of the second and third column in the list
        for row in reader:
            column2_data, column3_data, is_duplicate = row[3], row[4], int(row[5])
            data_list.append((column2_data, column3_data, is_duplicate))
    # print(data_list)
    return data_list


questions = getQuestions()
countLine = 0
Hit = 0
Hit2 = 0
Hit3 = 0
q1q2markHit = 0
q1q2markHit2 = 0
q1q2markHit3 = 0
q1q2markHitActuallt = 0
start_time = time.time()
csvLists = []
questions_subset=questions[370000: 380000]
print(questions_subset)
print(len(questions_subset))
for question in questions_subset:
    sentences = [question[0], question[1]]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    csvList = [question[0].encode('utf-8').decode('utf-8'), question[1].encode('utf-8').decode('utf-8'),question[2]]

    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # 处理模型输出，获取预测结果
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    predictions=cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()
    print("预测结果:", predictions)
    csvList.append(predictions)
    csvLists.append(csvList)
    if question[2] == 1:
        q1q2markHitActuallt = q1q2markHitActuallt + 1
    if predictions >0.92:
        Hit = Hit + 1
        if question[2] == 1:
            q1q2markHit = q1q2markHit + 1
    countLine = countLine + 1
    if countLine > 1000:
        break
print("Time consuming: {:.2f}s".format(time.time() - start_time))
print("countLine:", countLine)
print("q1q2markHit:", q1q2markHit)
print("Hit:", Hit)
print("q1q2markHitActuallt:", q1q2markHitActuallt)
csv_file_path = 'data.csv'


# 写入CSV文件
with open("./csdModelOutput.csv", mode='w',encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csvLists)

print(f'Data has been written to {csv_file_path}')







# output_sentences = ['1']
# reference_sentences = ['2']
# start_time = time.time()
# similarity_score = ssem.evaluate(output_sentences, reference_sentences, n_jobs=1, level='sentence', output_format='mean')
# print("Time consuming: {:.2f}s".format(time.time() - start_time))
# print("Similarity score: ", similarity_score)
# pearson 越靠近0越好
