import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

# -------------------------- 1. 数据准备 --------------------------
# 加载数据集，指定分隔符为制表符，并无表头
dataset = pd.read_csv("ecommerce_dataset.csv", sep="\t", header=None)
print(f"\n数据集大小: {len(dataset)} 条")
print(f"\n前10条样本:")
print(dataset.head(10))

# 初始化并拟合标签编码器，将文本标签（如"电子产品"）转换为数字标签（如0, 1, 2...）
lbl = LabelEncoder()
lbl.fit(dataset[1].values)

# 查看类别分布
print(f"\n类别分布:")
print(dataset[1].value_counts())

# 显示标签映射
print(f"\n标签映射:")
for idx, label in enumerate(lbl.classes_):
    print(f"{idx}: {label}")

# 将数据按8:2的比例分割为训练集和测试集
# stratify 参数确保训练集和测试集中各类别的样本比例与原始数据集保持一致
x_train, x_test, train_label, test_label = train_test_split(
    list(dataset[0].values),  # 使用全部数据
    lbl.transform(dataset[1].values),
    test_size=0.2,
    stratify=dataset[1].values,
    random_state=42
)

print(f"\n训练集大小: {len(x_train)}")
print(f"测试集大小: {len(x_test)}")

# 加载BERT预训练的分词器（Tokenizer）
# 分词器负责将文本转换为模型可识别的输入ID、注意力掩码等
print("\n加载 BERT 分词器...")
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')

# 对训练集和测试集的文本进行编码
# truncation=True：如果句子长度超过max_length，则截断
# padding=True：将所有句子填充到max_length
# max_length=64：最大序列长度
print("编码训练集和测试集...")
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)


# -------------------------- 2. 数据集和数据加载器 --------------------------
# 自定义数据集类，继承自PyTorch的Dataset
# 用于处理编码后的数据和标签，方便后续批量读取
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 获取单个样本的方法
    def __getitem__(self, idx):
        # 从编码字典中提取input_ids, attention_mask等，并转换为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 添加标签，并转换为张量
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    # 返回数据集总样本数的方法
    def __len__(self):
        return len(self.labels)


# 实例化自定义数据集
train_dataset = NewsDataset(train_encoding, train_label) # 单个样本读取的数据集
test_dataset = NewsDataset(test_encoding, test_label)

# 使用DataLoader创建批量数据加载器
# batch_size=16：每个批次包含16个样本
# shuffle=True：在每个epoch开始时打乱数据，以提高模型泛化能力
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # 批量读取样本
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_dataloader)}")

# -------------------------- 3. 模型和优化器 --------------------------
# 加载BERT用于序列分类的预训练模型
# num_labels：指定分类任务的类别数量
print(f"\n加载 BERT 分类模型 (类别数: {len(lbl.classes_)})...")
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=len(lbl.classes_))

# 设置设备，优先使用CUDA（GPU），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 将模型移动到指定的设备上
model.to(device)

# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# -------------------------- 4. 训练和验证函数 --------------------------
# 定义训练函数
def train(epoch):
    # 设置模型为训练模式
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)

    # 遍历训练数据加载器
    for batch in train_loader:
        # 清除上一轮的梯度
        optim.zero_grad()

        # 将批次数据移动到指定设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 执行前向传播，得到损失和logits
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 自动计算损失
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向传播计算梯度
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optim.step()

        iter_num += 1
        # 每5步打印一次训练进度
        if (iter_num % 5 == 0):
            print("Epoch: %d, iter: %d/%d, loss: %.4f, progress: %.2f%%" % (
                epoch, iter_num, total_iter, loss.item(), iter_num / total_iter * 100))

    # 打印平均训练损失
    avg_loss = total_train_loss / len(train_loader)
    print("Epoch: %d, Average training loss: %.4f" % (epoch, avg_loss))
    return avg_loss


# 定义验证函数
def validation():
    # 设置模型为评估模式
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    # 遍历测试数据加载器
    for batch in test_dataloader:
        # 在验证阶段，不计算梯度
        with torch.no_grad():
            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 执行前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        # 将logits和标签从GPU移动到CPU，并转换为numpy数组
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # 计算平均准确率
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    avg_val_loss = total_eval_loss / len(test_dataloader)
    print("Validation Accuracy: %.4f" % (avg_val_accuracy))
    print("Average validation loss: %.4f" % (avg_val_loss))
    print("-" * 60)
    return avg_val_accuracy


# -------------------------- 5. 主训练循环 --------------------------
# 循环训练4个epoch
NUM_EPOCHS = 4
print("\n" + "=" * 60)
print(f"开始训练 (共 {NUM_EPOCHS} 个 epochs)")
print("=" * 60 + "\n")

best_accuracy = 0
for epoch in range(NUM_EPOCHS):
    print(f"{'=' * 60}")
    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'=' * 60}")
    # 训练模型
    train(epoch)
    # 验证模型
    val_acc = validation()

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        print(f"新的最佳准确率: {best_accuracy:.4f}")

print("\n" + "=" * 60)
print(f"训练完成！最佳验证准确率: {best_accuracy:.4f}")
print("=" * 60)


# -------------------------- 6. 测试新样本 --------------------------
print("\n" + "=" * 60)
print("测试新样本分类效果")
print("=" * 60)

def predict_sample(text):
    """预测单个文本样本的类别"""
    # 设置模型为评估模式
    model.eval()

    # 编码文本
    encoding = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors='pt')

    # 移动到设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 获取预测结果和概率
    probs = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    # 转换为标签名称
    predicted_label = lbl.classes_[predicted_class.item()]
    confidence_score = confidence.item()

    # 获取所有类别的概率
    all_probs = probs[0].cpu().numpy()

    return predicted_label, confidence_score, all_probs


# 测试样本列表（包含4个类别的样本）
test_samples = [
    # 电子产品
    "这款笔记本电脑性能强劲，运行速度很快",
    # 服装
    "这件衣服面料舒适，穿着很透气",
    # 食品
    "这个零食很好吃，味道香甜可口",
    # 图书
    "这本书内容丰富，讲解详细",
]

print("\n开始测试样本...")
correct_predictions = 0
total_predictions = len(test_samples)

for i, text in enumerate(test_samples, 1):
    predicted_label, confidence, all_probs = predict_sample(text)

    print(f"\n{'=' * 60}")
    print(f"测试样本 {i}/{total_predictions}")
    print(f"{'=' * 60}")
    print(f"输入文本: {text}")
    print(f"预测类别: {predicted_label}")
    print(f"置信度: {confidence:.4f}")

    # 显示所有类别的概率分布
    print(f"\n所有类别的概率:")
    sorted_indices = all_probs.argsort()[::-1]
    for idx in sorted_indices:
        print(f"  {lbl.classes_[idx]}: {all_probs[idx]:.4f}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
