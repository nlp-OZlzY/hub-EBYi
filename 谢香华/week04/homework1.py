import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BerTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datsets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df=pd.read_csv("F:/BADOU_AI/HOME_WORK/dataset.csv",seq=",",header=None)
print(dataset_df)

# 初始化labelEncoder，用于将文本标签转化为数字标签
lbl=LabelEncoder()
labels=lbl.fit_transform(dataset_df[1].values[:500])
texts=list(dataset_df[0].values[:500])

# 分割数据为训练集和测试机
x_train,x_test,train_labels,test_labels=train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

tokenizer=BerTokenizer.from_pretrained('../models/google-bert/bert-base-chinese')
model=BertForSequenceClassification.from_pretrained('../models/google-bert/bert-base-chinese',num_labels=17)

train_encodings=tokenizer(x_train,truncation=True,padding=True,max_length=64)
test_encodings=tokenizer(x_test,truncation=True,padding=True,max_length=64)

train_dataset=Dataset.from_dict({
    'input_ids':train_encodings['input_ids'],
    'attention_mask':train_encodings['attention_mask'],
    'labels':train_labels
})

test_dataset=Dataset.from_dict({
    'input_ids':test_encodings['input_ids'],
    'attention_mask':test_encodings['attention_mask'],
    'labels':test_labels
})


def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    return {'accuracy':(predictions==labels).mean()}


training_args=TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()



