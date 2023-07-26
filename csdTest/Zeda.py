import os
import sys
import json
sys.path.insert(1, os.path.realpath('..'))
import requests
from typing import Any

# Set LLM = "qwen-plus-internal-v1" if try to use anolis copilot with "通义千问".
# In other cases, you can set the name of the LLM according to your preference.
LLM = "qwen-plus-internal-v1"

# API key for calling "通义千问".
DSAHSCOPE_KEY = "4e6k4p6bi5pqUEucVR63yWIqht62jC5w72807B8FE3FC11EDAA072AEC6FC183A8"

# HOST and PORT are needed if the backend is a personal llm.
HOST = "47.92.68.166"
PORT = "5000"

if LLM == "qwen-plus-internal-v1":
    try:
        from dashscope import Generation
    except ImportError:
        print('dashscope is not available, try pip install dashscope')
        raise ImportError('dashscope is not available')


class LM:
    def __init__(self, llm=LLM, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.llm = llm

    def call(self, input_prompt: str) -> Any:
        """
        Call the language model API.

        Args:
            input_prompt: The input prompt.

        Returns:
            The response from the API.
        """
        if self.llm == "qwen-plus-internal-v1":
            chat = Generation()
            req = chat.call(model=LLM, api_key=DSAHSCOPE_KEY, prompt=input_prompt, top_p=0.3)
            return req
        else:
            headers = {'content-type': 'application/json'}
            request_param = {
                "prompt": input_prompt,
            }
            try:
                req = requests.post(f"http://{str(self.host)}:{str(self.port)}", json=request_param, headers=headers)
                return req
            except Exception:
                print("Error: LLM connection fails. Please check the configuration for llm or your server status!")
                return None
csdlm=LM()
while(1):
    csdstr=input("请输入问题：")
    strcsd = csdlm.call(csdstr)
    print(strcsd["output"]["text"])

