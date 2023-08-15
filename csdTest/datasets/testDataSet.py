from datasets import Dataset
csv_file_path = r"E:\GPTCache\问题时间敏感分析\quora-question-pairs\train.csv"
# 读取数据集
dataset = Dataset.from_csv(csv_file_path)
datasetTest = dataset.select(range(370000, 404200 + 1))
dataset = dataset.select(range(360000))
print(dataset)
# Dataset({
#     features: ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
#     num_rows: 404290
# })
#熟米饭25%的碳水
#蛋白质 碳水的用途，热量摄入充足，供力量训练的力量
#加载token 加载 DistilBERT 分词器来预处理该text字段
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained( "distilbert-base-uncased" )

#创建一个预处理函数来对text序列进行标记和截断，使其长度不超过 DistilBERT 的最大输入长度
def preprocess_function(examples):
    question1 = (examples["question1"])
    question2 = (examples["question2"])
    print(question2)
    # x=examples["question1"]+examples["question2"]
    return tokenizer(text=question1,text_pair=question2, truncation=True)
#要将预处理函数应用于整个数据集，请使用 🤗 数据集映射函数。map您可以通过设置batched=True一次处理数据集的多个元素来加快速度：

tokenized_imdb = dataset.map(preprocess_function, batched=False)
tokenized_imdbtest = datasetTest.map(preprocess_function, batched=False)
print(tokenized_imdb)
print(tokenized_imdbtest)
#现在使用DataCollatorWithPadding创建一批示例。在整理过程中动态地将句子填充到批次中的最长长度比将整个数据集填充到最大长度更有效
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#在训练期间包含指标通常有助于评估模型的性能,评估方法
import evaluate
accuracy = evaluate.load("accuracy",cache_dir="..\..\csdTest")

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
    "distilbert-base-uncased", cache_dir="..\..\csdTest",num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="..\csdTest",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb,
    eval_dataset=tokenized_imdbtest,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()