import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from gpt3_api import make_requests_35 as make_gpt35_requests
from bs4 import BeautifulSoup 
import requests 
import json
#设置参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="web_finetune_data.jsonl"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://zamesin.me/clubhouse-elon-musk-interview"
        # default='https://baike.baidu.com/item/%E5%88%80%E9%83%8E/35482?fr=ge_ala'
    )
    return parser.parse_args()
#设置基础信息
api_key = "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"
engine = "gpt-3.5-turbo"
#删除以下段落中的 html 标签，仅保留文本。
instruct_clean = "Remove html tags in the following paragraph, leave only text.\n"
#将英文翻译成中文，仅使用简体字。 如果需要，去除前缀，例如“输出”。\n
instruct_translate = "Translate english into chinese, use simplified chinese character only. Strip prefix such as \'output\', if needed.\n"

if __name__ == '__main__':
    args = parse_args()
    url = args.url
    print("!!!!!", url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    print("================================================")
    print("paragraphs:",paragraphs)
    print("================================================")
    #clean data
    cleanpragraphs = []
    #获取段落中内容
    for i, p in enumerate(paragraphs):
        prompts = instruct_clean + p.text
        results = make_gpt35_requests(
            engine=engine,
            prompts=prompts,
            max_tokens=350,
            temperature=0,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=1.5,
            stop_sequences=[f"Example {3}", "Task:"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=api_key)
        text = results[0]["response"]["choices"][0]["message"]["content"].strip()
        print("text:",text)
        cleanpragraphs.append(text)

    #crawl dialogue
    speaker = ""
    Question = ""
    Answer = ""
    qalist = []
    qa = {}
    #处理数据
    for paragraph in cleanpragraphs:
        strlist = paragraph.split(":")
        speaker_curr = strlist[0].strip()
        text_curr = ''.join(strlist[1:]).strip()
        l = len(strlist)

        if speaker != '':
            if speaker_curr != 'Elon Musk':
                if Question != '' and Answer != '':
                    qa['Q'] = Question
                    qa['A'] = Answer
                    qalist.append(qa)
                    qa = {}
                    Question = ""
                    Answer = ""

                if l == 1:
                    Question = Question + text_curr
                else:
                    if speaker_curr != speaker:
                        Question = text_curr
                    else:
                        Question = Question + text_curr
                    speaker = speaker_curr
            elif speaker_curr == 'Elon Musk':
                Answer = text_curr
        else:
            if l > 1:
                speaker = speaker_curr
                if speaker != 'Elon Musk':
                    Question = text_curr
    
    #translate to chinese
    for i, qa in enumerate(qalist):
        instruct = instruct_translate   #instruct = instruct_translate
        qtext = "Input: " + qa['Q']
        atext = "Input: " + qa['A']

        prompts = instruct + qtext
        results = make_gpt35_requests(
            engine=engine,
            prompts=prompts,
            max_tokens=350,
            temperature=0,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=1.5,
            stop_sequences=[f"Example {3}", "Task:"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=api_key)
        qalist[i]['Q'] = results[0]["response"]["choices"][0]["message"]["content"].strip()
        tmp = qalist[i]['Q'] .split('：')[0]
        if tmp == '回答' or tmp == '输出':
            qalist[i]['Q']  = "".join(tmp[1:])

        prompts = instruct + atext
        results = make_gpt35_requests(
            engine=engine,
            prompts=prompts,
            max_tokens=350,
            temperature=0,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=1.5,
            stop_sequences=[f"Example {3}", "Task:"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=api_key)
        anwser = results[0]["response"]["choices"][0]["message"]["content"]

        qalist[i]['A'] = anwser.strip()
        tmp = qalist[i]['A'] .split('：')[0]
        if tmp == '回答' or tmp == '输出':
            qalist[i]['A']  = "".join(tmp[1:])
        print(qalist[i])

    #dump to json as finetune data
    jsonlist = []
    history_len = 1
    for i, qa in enumerate(qalist):
        dataslot = {}
        dataslot['instruction'] = qa['Q']
        dataslot['input'] = ""
        dataslot['output'] = qa['A']
        
        history = []
        if i >= history_len:
            start = i - history_len
            for j in range(start, i):
                history.append([qalist[j]['Q'], qalist[j]['A']])
        dataslot['history'] = history
        
        jsonlist.append(dataslot)

    with open(os.path.join(args.batch_dir, args.output_file), 'w') as fin:        
        json.dump(jsonlist, fin, ensure_ascii=False, indent=4)



