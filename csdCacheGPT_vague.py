import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

from gptcache import cache
from gptcache.adapter import openai
from gptcache.adapter import qwen
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import os
os.environ["OPENAI_API_KEY"] = ""
print("Cache loading.....")

onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

questions = [
    "你认为我能成功吗?",
    "什么是github？",
    "你好",
    "你知道上海吗?",
    "你知道北京吗?"
]
# "do you think i can success?",
# "what is github？",
# "hello",
# "do you know shanghai?",
# "do you know beijing?"

for question in questions:
    start_time = time.time()
    response = qwen.ChatCompletion.create(
        model='gpt-3.5-turbo',  #调用的千问的模型
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
