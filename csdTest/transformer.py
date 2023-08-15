from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

# 用于测试预训练模型，没啥用
model_path = r"../pretrained_model/IDEA-CCNL(Erlangshen-Roberta-110M-Sentiment)"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("今天心情很好")
print(result)
