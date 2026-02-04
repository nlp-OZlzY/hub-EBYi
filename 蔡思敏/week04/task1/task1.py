"""
@Author  :  CAISIMIN
@Date    :  2026/2/2 22:31

对文本进行情感分类
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch

# 查看所有的列
all_columns = pd.read_csv("sentimentdataset.csv", nrows=0).columns.to_list();
# print(all_columns)

# 读取text和Sentiment
datasets = pd.read_csv("sentimentdataset.csv", sep=",", usecols=['Text', 'Sentiment'])
# print(datasets.head(1))

# 处理label,去除空格
sentiment_label = datasets['Sentiment'].apply(lambda x: x.strip())
print(len(set(sentiment_label)))

lbl = LabelEncoder()
labels = lbl.fit_transform(sentiment_label.values)
print(type(labels))  # <class 'numpy.ndarray'>

texts = list((datasets['Text'].apply(lambda x: x.strip())))
print(type(texts))  # <class 'list'>

x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    shuffle=True
)

tokenizers = BertTokenizer.from_pretrained("../Week03/01-intent-classify/assets/models/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("../Week03/01-intent-classify/assets/models/bert-base-chinese", num_labels=191)

train_encodings = tokenizers(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizers(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test
})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'acc': (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# 新样本
new_text = "这部电影真是太棒了，剧情紧凑，演员演技在线！"

# 加载分词器和模型
tokenizers = BertTokenizer.from_pretrained("../Week03/01-intent-classify/assets/models/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("./results")

# 编码新样本
inputs = tokenizers(new_text, return_tensors="pt", truncation=True, padding=True, max_length=64)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 预测
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

# 解码预测结果
predicted_label = lbl.inverse_transform([predicted_class])[0]
print(f"预测的情感类别: {predicted_label}")
