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
os.environ["OPENAI_API_KEY"] = "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"
print("Cache loading.....")

qianWen = QianWenEmbedding()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=qianWen.dimension))
cache.init(
    embedding_func=qianWen.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()


def getQuestions():
    data_list = []
    csv_file_path = 'test.csv'

    # Open the CSV file in read mode
    with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Skip the header row (first row)
        next(reader)

        # Iterate over each row and store the values of the second and third column in the list
        for row in reader:
            column2_data, column3_data = row[1], row[2]
            data_list.append(column2_data)
            data_list.append(column3_data)
    print(data_list)
    return data_list

questions = getQuestions()
print(questions)
# questions = [
#     "你认为我能成功吗?",
#     "什么是github？",
#     "你好",
#     "你知道上海吗?",
#     "你知道北京吗?",
#     "你是谁呀?",
#     "你明不明白我说的话"
# ]
# "do you think i can success?",
# "what is github？",
# "hello",
# "do you know shanghai?",
# "do you know beijing?"

count=0;
for question in questions:
    start_time = time.time()
    response = qwen.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    if isinstance(response, dict):
        count=count+1
    print(f'Answer: {(response)}\n')