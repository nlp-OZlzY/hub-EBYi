import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# ================ 定义不同复杂度的模型 ================

class SimpleClassifier1(nn.Module):
    """模型1: 最简单的1层网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)  # 直接映射，没有隐藏层

    def forward(self, x):
        return self.fc1(x)


class SimpleClassifier2(nn.Module):
    """模型2: 2层网络（原始版本）"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SimpleClassifier3(nn.Module):
    """模型3: 3层网络，隐藏层节点较少"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 添加dropout防止过拟合

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out


class SimpleClassifier4(nn.Module):
    """模型4: 3层网络，隐藏层节点较多"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        return out


class SimpleClassifier5(nn.Module):
    """模型5: 4层深度网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier5, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        return out


# ================ 训练和对比函数 ================

def train_model(model, model_name, dataloader, criterion, optimizer, num_epochs=10):
    """训练单个模型并记录loss"""
    print(f"\n{'=' * 50}")
    print(f"开始训练模型: {model_name}")
    print(f"模型结构: {model}")

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1:2d}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return epoch_losses


# ================ 主程序 ================

# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义训练参数
hidden_dim = 128
output_dim = len(label_to_index)

# 定义要对比的模型
models_config = [
    {"name": "1层网络(无隐藏层)", "model_class": SimpleClassifier1, "hidden_dim": hidden_dim},
    {"name": "2层网络(原始)", "model_class": SimpleClassifier2, "hidden_dim": hidden_dim},
    {"name": "3层网络(节点较少)", "model_class": SimpleClassifier3, "hidden_dim": hidden_dim},
    {"name": "3层网络(节点较多)", "model_class": SimpleClassifier4, "hidden_dim": hidden_dim},
    {"name": "4层深度网络", "model_class": SimpleClassifier5, "hidden_dim": hidden_dim},
]

# 存储每个模型的loss历史
all_losses = {}
all_models = {}

# 训练所有模型
num_epochs = 15

for config in models_config:
    # 创建模型
    model = config["model_class"](vocab_size, config["hidden_dim"], output_dim)

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    losses = train_model(
        model,
        config["name"],
        dataloader,
        criterion,
        optimizer,
        num_epochs
    )

    # 保存结果
    all_losses[config["name"]] = losses
    all_models[config["name"]] = model

# ================ 可视化对比结果 ================
import matplotlib.pyplot as plt
import warnings

# 方法1：设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure(figsize=(15, 10))

# 子图1: 所有模型的loss曲线对比
plt.subplot(2, 2, 1)
for model_name, losses in all_losses.items():
    plt.plot(range(1, num_epochs + 1), losses, marker='o', label=model_name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss变化对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 最终loss对比(条形图)
plt.subplot(2, 2, 2)
final_losses = {name: losses[-1] for name, losses in all_losses.items()}
names = list(final_losses.keys())
values = list(final_losses.values())

bars = plt.bar(names, values, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.xlabel('模型结构')
plt.ylabel('最终Loss')
plt.title('各模型最终Loss值对比')
plt.xticks(rotation=45, ha='right')

# 在条形上添加数值
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 子图3: 收敛速度对比(前5个epoch)
plt.subplot(2, 2, 3)
early_epochs = 5
for model_name, losses in all_losses.items():
    plt.plot(range(1, early_epochs + 1), losses[:early_epochs],
             marker='s', label=model_name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('前5个Epoch的Loss变化(收敛速度)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图4: 参数数量与最终loss的关系
plt.subplot(2, 2, 4)
param_counts = []
for config in models_config:
    model = config["model_class"](vocab_size, config["hidden_dim"], output_dim)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_counts.append(param_count)

# 创建散点图
scatter = plt.scatter(param_counts, values, s=150, c=range(len(models_config)),
                      cmap='viridis', alpha=0.7)

# 添加标签
for i, (name, param, loss) in enumerate(zip(names, param_counts, values)):
    plt.annotate(name, (param, loss), xytext=(10, 5),
                 textcoords='offset points', fontsize=9)

plt.xlabel('参数数量')
plt.ylabel('最终Loss')
plt.title('参数数量 vs 最终Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================ 模型性能总结 ================

print("\n" + "=" * 60)
print("模型性能总结报告")
print("=" * 60)

print("\n{:<25} {:<15} {:<15} {:<15}".format(
    "模型结构", "最终Loss", "参数数量", "性能评级"
))
print("-" * 70)

# 按最终loss排序
sorted_results = sorted(
    zip(names, values, param_counts),
    key=lambda x: x[1]  # 按loss排序
)

for name, loss, params in sorted_results:
    # 简单评级
    if loss == min(values):
        rating = "★★★★★ (最佳)"
    elif loss < sum(values) / len(values):
        rating = "★★★★ (良好)"
    else:
        rating = "★★★ (一般)"

    print("{:<25} {:<15.4f} {:<15,} {:<15}".format(
        name, loss, params, rating
    ))

print("\n" + "=" * 60)
print("关键发现:")
print("1. 网络层数增加通常会使loss降低，但可能出现过拟合风险")
print("2. 参数数量不是越多越好，需要平衡模型容量和泛化能力")
print("3. 添加适当的dropout可以帮助防止过拟合")
print("4. 收敛速度与网络复杂度不一定正相关")
print("=" * 60)


# ================ 使用最佳模型进行预测 ================

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """使用指定模型进行分类"""
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 选择最佳模型（loss最低的）
best_model_name = sorted_results[0][0]
best_model = all_models[best_model_name]

print(f"\n使用最佳模型 '{best_model_name}' 进行预测:")

index_to_label = {i: label for label, i in label_to_index.items()}

# 测试一些样本
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌",
    "设置明天早上7点的闹钟",
    "今天上海温度多少"
]

for test_text in test_texts:
    predicted_class = classify_text(test_text, best_model, char_to_index,
                                    vocab_size, max_len, index_to_label)
    print(f"输入 '{test_text}' → 预测为: '{predicted_class}'")