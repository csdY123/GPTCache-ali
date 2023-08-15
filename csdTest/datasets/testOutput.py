from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 载入模型
model = AutoModelForSequenceClassification.from_pretrained("..\csdTest\checkpoint-45000")

# 载入tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 用于测试的文本输入
text1 = "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?"
text2 = "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"

# 使用tokenizer将输入文本转换为模型可接受的格式
inputs = tokenizer(text1, text2, truncation=True, padding=True, return_tensors="pt")

# 使用模型进行预测
with torch.no_grad():
    outputs = model(**inputs)

# 处理模型输出，获取预测结果
predictions = outputs.logits.argmax(dim=-1)

print("预测结果:", predictions.item())
print(predictions)