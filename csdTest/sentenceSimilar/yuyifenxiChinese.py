from transformers import BertTokenizer, BertForNextSentencePrediction

sentence1 = "你知道上海吗"
sentence2 = "你知道上海吗"
combined_sentence = "你知道上海吗"
print(combined_sentence)
# 加载预训练的 BERT 模型和分词器
model_name = "bert-base-chinese"  # 你可以选择适合中文的 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_name,cache_dir="..\csdTest")
model = BertForNextSentencePrediction.from_pretrained(model_name,cache_dir="..\csdTest")

# 准备数据
encoded_text = tokenizer.encode_plus(
    combined_sentence,
    add_special_tokens=True,
    max_length=128,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)

# 进行推理
outputs = model(**encoded_text)
print(outputs)
logits = outputs.logits
print(logits)
probs = logits.softmax(dim=-1)
print(probs)
is_next_sentence = bool(probs[0][0] > probs[0][1])  # 判断是否为下一句

# 输出结果
if is_next_sentence:
    print("两个句子语义一致。")
else:
    print("两个句子语义不一致。")
