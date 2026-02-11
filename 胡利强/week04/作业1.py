import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

# -------------------------- 1. 数据准备（核心修正） --------------------------
# 修正1：正确加载数据集（分隔符为逗号，跳过表头，指定列名）
dataset = pd.read_csv(
    "../Week01/ChnSentiCorp_htl_all.csv",
    sep=",",  # 原数据是逗号分隔，而非\t
    header=0,  # 第一行是表头，跳过
    names=["label", "review"]  # 自定义列名，方便后续索引
)
print("数据集前5行：")
print(dataset.head())

# 修正2：检查并处理空值（避免后续报错）
dataset = dataset.dropna(subset=["label", "review"])

# 初始化并拟合标签编码器（现在用正确的列名索引）
lbl = LabelEncoder()
lbl.fit(dataset["label"].values)
# 保存标签映射（方便后续预测解析）
label_mapping = dict(zip(range(len(lbl.classes_)), lbl.classes_))

# 修正3：分割数据时使用正确的列名，且只取前500条样本
x_train, x_test, train_label, test_label = train_test_split(
    list(dataset["review"].values[:500]),  # 文本列
    lbl.transform(dataset["label"].values[:500]),  # 标签列
    test_size=0.2,
    stratify=dataset["label"][:500].values  # 保持类别分布一致
)

# 加载BERT中文分词器
tokenizer = BertTokenizer.from_pretrained('../models/google-bert/bert-base-chinese')

# 文本编码（参数保持不变）
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)


# -------------------------- 2. 数据集和数据加载器（无修改） --------------------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# -------------------------- 3. 模型和优化器（微调num_labels） --------------------------
# 修正4：num_labels自动匹配数据集类别数，避免硬编码错误
model = BertForSequenceClassification.from_pretrained(
    '../models/google-bert/bert-base-chinese',
    num_labels=len(lbl.classes_)  # 替代原硬编码的17
)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# -------------------------- 4. 精度计算、训练、验证函数（无修改） --------------------------
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)

    for batch in train_loader:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")


# -------------------------- 5. 新增：单样本预测函数（验证分类效果） --------------------------
def predict_single_sample(text):
    """输入单个文本样本，返回分类结果和概率"""
    # 文本编码（与训练保持一致）
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )

    # 数据移到设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)  # 转为概率

    # 解析结果
    pred_idx = torch.argmax(probs, dim=1).cpu().item()
    pred_label = label_mapping[pred_idx]
    pred_prob = probs[0][pred_idx].cpu().item()

    return pred_label, pred_prob


# -------------------------- 6. 主训练循环 --------------------------
for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()

# -------------------------- 7. 测试新样本（验证分类效果） --------------------------
print("\n========== 测试新样本 ==========")
test_text = input("请输入待分类的酒店评论：")
pred_label, pred_prob = predict_single_sample(test_text)
print(f"\n预测结果：")
print(f"分类标签（情感）：{pred_label}")
print(f"预测概率：{pred_prob:.4f}")
