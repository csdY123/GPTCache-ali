# -*- coding: utf-8 -*-
import csv

from SsemModel import SemanticSimilarity
import time

ssem = SemanticSimilarity(model_name='bert-base-multilingual-cased', metric='cosine', custom_embeddings=None)


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
    print(data_list)
    return data_list


questions = getQuestions()
countLine = 0
Hit = 0
Hit2 = 0
Hit3 = 0
Hit4 = 0
q1q2markHit = 0
q1q2markHit2 = 0
q1q2markHit3 = 0
q1q2markHit4 = 0
q1q2markHitActuallt = 0
start_time = time.time()
csvLists = []

for question in questions:
    csvList = [question[0].encode('utf-8').decode('utf-8'), question[1].encode('utf-8').decode('utf-8'),question[2]]
    output_sentences = [question[0]]
    reference_sentences = [question[1]]
    similarity_score = ssem.evaluate(output_sentences, reference_sentences, n_jobs=1, level='token',
                                     output_format='mean')
    csvList.append(similarity_score)
    csvLists.append(csvList)
    if question[2] == 1:
        q1q2markHitActuallt = q1q2markHitActuallt + 1
    if similarity_score > 0.55:
        Hit = Hit + 1
        if question[2] == 1:
            q1q2markHit = q1q2markHit + 1
    if similarity_score > 0.60:
        Hit2 = Hit2 + 1
        if question[2] == 1:
            q1q2markHit2 = q1q2markHit2 + 1
    if similarity_score > 0.65:
        Hit3 = Hit3 + 1
        if question[2] == 1:
            q1q2markHit3 = q1q2markHit3 + 1
    if similarity_score > 0.50:
        Hit4 = Hit4 + 1
        if question[2] == 1:
            q1q2markHit4 = q1q2markHit4 + 1
    print("Similarity score: ", similarity_score)
    countLine = countLine + 1
    if countLine > 1000:
        break
print("Time consuming: {:.2f}s".format(time.time() - start_time))
print("countLine:", countLine)
print("q1q2markHit:", q1q2markHit)
print("q1q2markHit2:", q1q2markHit2)
print("q1q2markHit3:", q1q2markHit3)
print("q1q2markHit3:", q1q2markHit4)
print("Hit:", Hit)
print("Hit2:", Hit2)
print("Hit3:", Hit3)
print("Hit3:", Hit4)
print("q1q2markHitActuallt:", q1q2markHitActuallt)
csv_file_path = 'data.csv'


# 写入CSV文件
# with open("./csdCsvCosinelsi.csv", mode='w',encoding='utf-8', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(csvLists)
#
# print(f'Data has been written to {csv_file_path}')







# output_sentences = ['1']
# reference_sentences = ['2']
# start_time = time.time()
# similarity_score = ssem.evaluate(output_sentences, reference_sentences, n_jobs=1, level='sentence', output_format='mean')
# print("Time consuming: {:.2f}s".format(time.time() - start_time))
# print("Similarity score: ", similarity_score)
# pearson 越靠近0越好
