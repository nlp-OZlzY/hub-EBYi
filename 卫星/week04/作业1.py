import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate


dataset = load_dataset("C-MTEB/TNews-classification")

# 类别名：id -> name
label_names = dataset["train"].features["label"].names
num_labels = len(label_names)

print("类别数量:", num_labels)
print("类别列表:", label_names)

model_name = "models/google-bert/bert-base-chinese" 

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=64)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./tnews_bert_output",     # 输出目录
    num_train_epochs=10,                   
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",          # 每个 epoch 评估一次
    save_strategy="epoch",                # 每个 epoch 保存一次
    load_best_model_at_end=True,          # 训练结束加载最优模型
    metric_for_best_model="accuracy",
    logging_steps=100,
    fp16=torch.cuda.is_available(),       
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
eval_result = trainer.evaluate()
print("\n验证集评估结果:", eval_result)

def predict_one(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = trainer.model(**inputs).logits

    pred_id = int(torch.argmax(logits, dim=-1).cpu().item())
    pred_label = label_names[pred_id]
    return pred_id, pred_label

new_text = "中国女排在世界锦标赛中夺冠，球迷热议比赛表现"
pred_id, pred_label = predict_one(new_text)

print("\n【新样本预测】")
print("文本:", new_text)
print("预测类别ID:", pred_id)
print("预测类别名:", pred_label)
