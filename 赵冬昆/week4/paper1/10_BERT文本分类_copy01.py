import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df = pd.read_csv("../online_shopping_10_cats.csv")
# 👇 关键：先打乱整个数据集！
dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)
# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()

# 取前 N 条
N = 2000
subset_df = dataset_df.iloc[:N]
# 提取并确保是字符串
texts = subset_df['review'].astype(str).tolist()
labels = lbl.fit_transform(subset_df['cat'])
# # 拟合数据并转换前500个标签，得到数字标签
# labels = lbl.fit_transform(dataset_df['cat'].values[:N])
# # 提取前500个文本内容
# texts = list(dataset_df['review'].values[:N])

# 检查类别数
print(f"使用的类别数: {len(np.unique(labels))}")  # 应该是 10
print(f"各类别样本数:\n{pd.Series(labels).value_counts().sort_index()}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)




# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('../models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('../models/google-bert/bert-base-chinese', num_labels=10)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})





# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
    metric_for_best_model="accuracy",
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

# trainer 是比较简单，适合训练过程比较规范化的模型
# 如果我要定制化训练过程，trainer无法满足

# 训练完成后，显式保存最佳模型到 output_dir 根目录
trainer.save_model("./results")  # 👈 关键！这会生成 pytorch_model.bin

# ====================================================
# 新增：用训练好的模型预测新样本（推理）
# ====================================================

# 1. 保存 LabelEncoder（以便后续加载使用）
import joblib
joblib.dump(lbl, './results/label_encoder.pkl')

# 2. 加载最佳模型（Trainer 已自动保存在 output_dir）
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# 重新加载 tokenizer（和训练时一致）
tokenizer = BertTokenizer.from_pretrained('../models/google-bert/bert-base-chinese')
# 加载微调后的模型（自动加载 best model）
model = BertForSequenceClassification.from_pretrained('./results')
model.eval()  # 设置为评估模式

# 3. 加载标签编码器
lbl = joblib.load('./results/label_encoder.pkl')

# 4. 定义预测函数
def predict(text: str, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
    pred_label = lbl.inverse_transform([pred_id])[0]
    return pred_label, confidence

# 5. 测试新样本
print("\n🔍 开始测试新样本：")
test_samples = [
    "这款手机拍照特别清晰，电池也很耐用！",
    "苹果很新鲜，就是有点贵。",
    "书的内容很有深度，值得反复阅读。",
    "热水器安装后一直漏水，客服也不管。",
    "这件衣服尺码偏小，质量一般。"
]

for sample in test_samples:
    pred, conf = predict(sample)
    print(f"输入: {sample}")
    print(f"预测类别: {pred} (置信度: {conf:.2f})\n")