import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# 预训练模型导入
from modelscope import AutoTokenizer, AutoModelForSequenceClassification

# 加载和预处理数据
dataset = pd.read_csv("toutiao_cat_data.txt", sep="_!_", header=None)

# 打印前五行数据
print(dataset.head())

# 初始化labelEncoder
labelEncoder = LabelEncoder()
train_data_num = 100000
# 拟合数据并转换标签，得到数字标签
print("标签统计:", set(dataset[2].values[:train_data_num]))
labels = labelEncoder.fit_transform(dataset[2].values[:train_data_num])

# 在创建数据集后，训练前添加这段代码
print("标签统计:", labels)
print(f"最小标签值: {min(labels)}")
print(f"最大标签值: {max(labels)}")
print(f"唯一标签值: {sorted(set(labels))}")
print(f"标签数量: {len(set(labels))}")

# 确保标签从0开始连续
unique_labels = sorted(set(labels))
label_map = {old: new for new, old in enumerate(unique_labels)}
mapped_labels = [label_map[label] for label in labels]

print(f"映射后的标签范围: 0 到 {len(unique_labels)-1}")

# 提取文本内容
texts = list(dataset[3].values[:train_data_num])

# 分割数据为训练集和验证集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2, # 测试集占20%
    stratify=labels # 保持标签比例，确保训练集和测试集的标签比例一致
    )

# 从预训练模型中加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese', num_labels=len(unique_labels))

# 使用分词器对训练集和测试集进行编码
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face 的 Dataset 对象
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,        # 保守起步
    gradient_accumulation_steps=8,        # 4×8=32 等效 Batch
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,                     # 按总步数 10 % 预热
    weight_decay=0.01,
    learning_rate=3e-5,                   # 核心参数别忘了
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",     # 必须指定
    greater_is_better=True,
    fp16=True,                            # 8 GB 卡再省 30 % 显存
)


# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 开始训练模型
trainer.train()
# 评估模型
trainer.evaluate()


best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"The best model is located at: {best_model_path}")
    torch.save(best_model.state_dict(), './results/bert.pt')
    print("Best model saved to results/bert.pt")
else:
    print("Could not find the best model checkpoint.")


