
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import warnings
warnings.filterwarnings('ignore')

"""
购物行为预测模型微调脚本
使用BERT模型对购物行为数据进行分类预测
预测目标：购买频率(Frequency of Purchases)
"""

# 加载和预处理购物行为数据集
dataset_df = pd.read_csv("./shopping_behavior_updated.csv")

print(f"数据集形状: {dataset_df.shape}")
print(f"列名: {list(dataset_df.columns)}")

# 选择预测目标 - 购买频率
target_column = 'Frequency of Purchases'

# 特征工程 - 组合多个特征创建文本输入
def create_combined_features(row):
    """
    将多个购物行为特征组合成一段描述性文本
    
    Args:
        row: 数据集的一行数据
    
    Returns:
        str: 组合后的特征文本
    """
    features = []
    
    # 基本信息
    features.append(f"Age {row['Age']} {row['Gender']}")
    
    # 商品信息
    features.append(f"purchased {row['Item Purchased']} in {row['Category']} category")
    
    # 地理位置
    features.append(f"from {row['Location']}")
    
    # 商品属性
    features.append(f"size {row['Size']} color {row['Color']}")
    
    # 时间季节
    features.append(f"during {row['Season']}")
    
    # 价格信息
    features.append(f"cost ${row['Purchase Amount (USD)']}")
    
    # 评分
    features.append(f"rated {row['Review Rating']} stars")
    
    # 购买历史
    features.append(f"previous purchases {row['Previous Purchases']}")
    
    # 支付方式
    features.append(f"paid by {row['Payment Method']}")
    
    # 折扣和订阅
    if row['Discount Applied'] == 'Yes':
        features.append("with discount")
    if row['Subscription Status'] == 'Yes':
        features.append("subscriber")
    
    return ", ".join(features)

# 创建组合特征文本
print("正在创建组合特征...")
dataset_df['combined_text'] = dataset_df.apply(create_combined_features, axis=1)

# 编码目标变量
print("正在编码目标变量...")
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df[target_column])
label_names = lbl.classes_
print(f"预测类别: {label_names}")

# 提取文本特征
texts = dataset_df['combined_text'].tolist()

# 分割数据集
print("正在分割数据集...")
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels,   # 确保训练集和测试集的标签分布一致
    random_state=42    # 设置随机种子以保证结果可重现
)




# 从预训练模型加载英文BERT分词器和模型
# 使用bert-base-uncased因为购物行为数据集是英文的
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_names))

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=128：最大序列长度（比原来更长以容纳更多特征）
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': torch.tensor(train_labels)                 # 对应的标签（转换为tensor）
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': torch.tensor(test_labels)
})





# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    """
    计算模型评估指标
    
    Args:
        eval_pred: 包含模型预测logits和真实标签的元组
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率
    accuracy = (predictions == labels).mean()
    
    # 计算加权平均的精确率、召回率和F1分数
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./shopping_results',     # 训练输出目录，用于保存模型和状态
    num_train_epochs=5,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=200,                    # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./shopping_logs',       # 日志存储目录
    logging_steps=50,                    # 每隔50步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
    metric_for_best_model="accuracy",   # 以准确率为标准选择最佳模型
    greater_is_better=True,              # 准确率越高越好
    learning_rate=2e-5,                  # 学习率
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

print("开始训练购物行为预测模型...")
# 开始训练模型
trainer.train()

# 在测试集上进行最终评估
print("正在评估模型性能...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 保存微调后的模型
print("正在保存模型...")
model.save_pretrained('./fine_tuned_shopping_model')
tokenizer.save_pretrained('./fine_tuned_shopping_model')

# 测试一些示例预测
print("\n=== 模型预测示例 ===")
test_examples = [
    "Age 25 Female purchased Dress in Clothing category from California, size M color Blue during Spring, cost $85, rated 4.5 stars, previous purchases 12, paid by Credit Card, subscriber",
    "Age 45 Male purchased Boots in Footwear category from Texas, size L color Black during Winter, cost $95, rated 3.8 stars, previous purchases 8, paid by PayPal, with discount",
    "Age 30 Female purchased Sunglasses in Accessories category from Florida, size S color Pink during Summer, cost $45, rated 4.2 stars, previous purchases 5, paid by Debit Card"
]

# 对示例进行编码和预测
for i, example in enumerate(test_examples, 1):
    inputs = tokenizer(example, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        print(f"示例 {i}: {example}")
        print(f"预测类别: {label_names[predicted_class]}")
        print(f"置信度: {confidence:.3f}")
        print("-" * 80)

print("\n购物行为预测模型微调完成！")
print(f"模型已保存到: ./fine_tuned_shopping_model/")
print(f"支持的预测类别: {list(label_names)}")
