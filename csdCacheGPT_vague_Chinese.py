import time
import csv


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


from gptcache import cache
from gptcache.adapter import openai
from gptcache.adapter import qwen
from gptcache.embedding.qianwenEmbedding import QianWenEmbedding
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import os

os.environ["OPENAI_API_KEY"] = ""
print("Cache loading.....")

qianWen = QianWenEmbedding()
# rocksdb、sqlite、mysql 可选
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=qianWen.dimension))
cache.init(
    embedding_func=qianWen.to_embeddings,  # 千问的embedding
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()


# 该函数用于加载csv数据集进行测试
def getQuestions():
    data_list = []
    csv_file_path = 'train.csv'

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


# questions = getQuestions()
# print(questions)
questions = [
    "你知道上海吗?",
    "你知道上海吗?",
]

start_time = time.time()
for question in questions:
    response = qwen.ChatCompletion.create(
        model='qwen-plus-internal-v1',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {(response)}\n')

# countLine=0
# q1Hit=0
# q2Hit=0
# q1q2markHit=0
# q1q2markHitActuallt=0
# for question in questions:
#     countLine=countLine+1
#     for i in range(2):
#         print(i)
#         if i==0:
#             print()
#             print()
#             print()
#             start_time = time.time()
#             response = qwen.ChatCompletion.create(
#                 model='gpt-3.5-turbo',
#                 messages=[
#                     {
#                         'role': 'user',
#                         'content': question[i]
#                     }
#                 ],
#             )
#             print(f'Question: {question}')
#             print("Time consuming: {:.2f}s".format(time.time() - start_time))
#             if isinstance(response, dict):
#                 q1Hit=q1Hit+1
#             print(f'Answer: {(response)}\n')
#             print()
#             print()
#             print()
#         else:
#             print()
#             print()
#             print()
#             start_time = time.time()
#             response = qwen.ChatCompletion.create(
#                 model='gpt-3.5-turbo',
#                 messages=[
#                     {
#                         'role': 'user',
#                         'content': question[i]
#                     }
#                 ],
#             )
#             print(f'Question: {question}')
#             print("Time consuming: {:.2f}s".format(time.time() - start_time))
#             if question[2] == 1:
#                 q1q2markHitActuallt = q1q2markHitActuallt + 1
#             if isinstance(response, dict):
#                 q2Hit=q2Hit+1
#                 if question[2]==1:
#                     q1q2markHit=q1q2markHit+1
#             print(f'Answer: {(response)}\n')
#             print()
#             print()
#             print()
#     if countLine>=1000:
#         break
# print("countLine:",countLine)
# print("q1Hit:",q1Hit)
# print("q2Hit:",q2Hit)
# print("q1q2markHit:",q1q2markHit)
# print("q1q2markHitActuallt:",q1q2markHitActuallt)
