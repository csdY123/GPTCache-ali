import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

print("Cache loading.....")

# To use GPTCache, that's all you need
# -------------------------------------------------
from gptcache import cache
from gptcache.adapter import openai
import os
os.environ["OPENAI_API_KEY"] = "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"

cache.init()
cache.set_openai_key()
# -------------------------------------------------

question = "真正的原因"
for _ in range(2):
    start_time = time.time()
    #openai.api_key = "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"
    response = openai.ChatCompletion.create(
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
    print(f'Answer: {response_text(response)}\n')

    #print("response是什么：",response)
    #{'gptcache': True, 'saved_token': [10, 116], 'choices': [{'message': {'role': 'assistant',
    #                                                                      'content': 'GitHub is a web-based platform used for version control and collaboration in software development projects. It provides a way for developers to store, manage, and track changes to their code. With GitHub, multiple people can work on the same project simultaneously, and any changes made are recorded and can be easily reviewed. Additionally, GitHub allows for easy collaboration and contribution, as developers can fork a project to make their own changes and then submit those changes to the original project for review and integration. It is widely used in the open-source community and is a popular platform for hosting and sharing code repositories.'},
    #                                                          'finish_reason': 'stop', 'index': 0}],
    # 'created': 1689317359, 'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
    # 'object': 'chat.completion'}