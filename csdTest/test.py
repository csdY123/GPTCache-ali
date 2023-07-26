from datasets import load_dataset
imdb = load_dataset("imdb",cache_dir="..\csdTest")
imdb["test"][0]
print(imdb)

#加载token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained( "distilbert-base-uncased" )

#创建一个预处理函数来对text序列进行标记和截断，使其长度不超过 DistilBERT 的最大输入长度
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy",cache_dir="..\csdTest")

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