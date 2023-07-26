import time
import openai
import os
openai.api_key="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"
def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


question = '北京在哪里?'

# OpenAI API original usage
start_time = time.time()
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