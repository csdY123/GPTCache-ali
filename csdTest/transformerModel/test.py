from datasets import load_dataset
imdb = load_dataset("imdb",cache_dir="..\csdTest")
imdb["test"][0]
print(imdb)
#熟米饭25%的碳水
#蛋白质 碳水的用途，热量摄入充足，供力量训练的力量
#加载token 加载 DistilBERT 分词器来预处理该text字段
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained( "distilbert-base-uncased" )

#创建一个预处理函数来对text序列进行标记和截断，使其长度不超过 DistilBERT 的最大输入长度
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
#要将预处理函数应用于整个数据集，请使用 🤗 数据集映射函数。map您可以通过设置batched=True一次处理数据集的多个元素来加快速度：

tokenized_imdb = imdb.map(preprocess_function, batched=True)
print(tokenized_imdb)
print(tokenized_imdb["train"][0])
#现在使用DataCollatorWithPadding创建一批示例。在整理过程中动态地将句子填充到批次中的最长长度比将整个数据集填充到最大长度更有效
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#在训练期间包含指标通常有助于评估模型的性能,评估方法
import evaluate
accuracy = evaluate.load("accuracy",cache_dir="..\csdTest")

#然后创建一个函数，将您的预测和标签传递给计算以计算准确性
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", cache_dir="..\csdTest",num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="..\csdTest",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()