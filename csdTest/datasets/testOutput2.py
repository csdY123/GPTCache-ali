from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline


model_path = r"..\csdTest\checkpoint-45000"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
text1 = "你知道北京吗？"
text2 = "你知道上海吗？"
input={
    "question1":text1,"question2":text2
}
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("今天心情很好")
print(result)