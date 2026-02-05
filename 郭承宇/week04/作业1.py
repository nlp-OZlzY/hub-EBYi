import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

dataset_1 = pd.read_json('E:\\ai八斗学院学习\\Data\\product-classification-hiring-demo\\train.jsonl',lines=True)

lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_1['category'].values[:1000])
texts = list(dataset_1['product_name'].values[:1000])

# import pdb;pdb.set_trace()

x_train,x_test,train_labels,test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained('E:\\ai八斗学院学习\\models\\google-bert\\bert-base-chinese')
model  = BertForSequenceClassification.from_pretrained('E:\\ai八斗学院学习\\models\\google-bert\\bert-base-chinese',num_labels=10)

train_encodings = tokenizer(x_train,truncation=True,padding=True,max_length=30)
test_encodings = tokenizer(x_test,truncation=True,padding=True,max_length=30)

# import pdb; pdb.set_trace()

train_dataset = Dataset.from_dict(
    {
        'input_ids':train_encodings['input_ids'],
        'attention_mask':train_encodings['attention_mask'],
        'labels':train_labels
    }
)

test_dataset = Dataset.from_dict(
    {
        'input_ids':test_encodings['input_ids'],
        'attention_mask':test_encodings['attention_mask'],
        'labels':test_labels
    }
)

def compute_metrics(pred):
    logits,labels = pred
    predictions = np.argmax(logits,axis=-1)
    return {'accuracy':(predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./homework',
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./homework_log',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

new_test = '世纪金徽四星酒42度250ml'
new_encodings = tokenizer(new_test,truncation=True,padding=True,max_length=30,return_tensors='pt')

model.eval()
with torch.no_grad():
    output = model(**new_encodings)

# import pdb; pdb.set_trace()

result_logits = output.logits
result_index = torch.argmax(result_logits,dim=-1).item()
result = lbl.inverse_transform([result_index])[0]
print(f'预测:{result}')


