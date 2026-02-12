import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ======================
# 1. 本地构造小型文本分类数据集（3 类）
# ======================
texts = [
    "我要订一张明天去北京的火车票",
    "帮我买一张机票",
    "我想预订酒店",
    "查询一下今天的天气",
    "帮我查一下订单状态",
    "我想查询余额",
    "你们这个服务太差了",
    "我要投诉客服",
    "对这次服务非常不满意"
]

labels = [
    0, 0, 0,   # 订票
    1, 1, 1,   # 查询
    2, 2, 2    # 投诉
]

label_map = {
    0: "订票意图",
    1: "查询意图",
    2: "投诉意图"
}

dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})

# ======================
# 2. 加载 tokenizer 和模型
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=3
)

# 冻结 BERT 主体（极大加速）
for param in model.bert.parameters():
    param.requires_grad = False

# ======================
# 3. 文本编码
# ======================
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ======================
# 4. 训练参数（作业级别）
# ======================
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    logging_steps=5,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# ======================
# 5. 训练
# ======================
trainer.train()

# ======================
# 6. 新样本预测（01 意图识别）
# ======================
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return label_map[pred_id]

test_text = "我想投诉一下这个订单"
print("输入文本：", test_text)
print("识别意图：", predict(test_text))
