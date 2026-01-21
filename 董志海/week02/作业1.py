"""
深度学习文本分类器

该脚本实现了基于字符级词袋模型（Bag of Words）的文本分类器。
主要功能包括：
- 从CSV文件加载文本数据
- 构建字符级词袋模型表示
- 使用简单的神经网络进行文本分类
- 对新文本进行分类预测
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 从CSV文件加载数据集，使用制表符分隔，无标题行
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 创建标签到索引的映射，用于将字符串标签转换为数值
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射表，包含填充字符'<pad>'作为索引0
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    """
    字符级词袋模型数据集类
    
    该类将文本转换为字符级词袋向量，用于神经网络训练。
    
    Args:
        texts: 输入的文本列表
        labels: 对应的数值标签列表
        char_to_index: 字符到索引的映射字典
        max_len: 文本的最大长度
        vocab_size: 词汇表的大小
    """

    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        创建字符级词袋向量
        
        遍历文本列表，将每个文本转换为固定长度的词袋向量。
        
        Returns:
            包含所有词袋向量的张量
        """
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
        """
        返回数据集的大小
        
        Returns:
            数据集中样本的数量
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取指定索引位置的数据样本
        
        Args:
            idx: 样本的索引
            
        Returns:
            词袋向量和对应标签的元组
        """
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    """
    简单的文本分类神经网络模型
    
    该模型包含一个输入层、一个隐藏层和一个输出层，使用ReLU激活函数。
    
    Args:
        input_dim: 输入层的维度
        hidden_dim: 隐藏层的维度
        output_dim: 输出层的维度（等于类别数量）
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入的特征张量
            
        Returns:
            模型的输出张量
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 创建数据集实例和数据加载器，用于批量训练
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义模型超参数并初始化模型、损失函数和优化器
hidden_dim = 128
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环：遍历多个epoch，在每个epoch中训练整个数据集
# epoch设置为1 ，即使节点数设置为15000，结果还是不符合预期的Loss: 2.4050
# epoch设置为10，节点设置为15，结果符合预期 Loss: 0.5855
num_epochs = 10
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
        if idx % 15 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对输入文本进行分类预测
    
    该函数将输入的文本转换为词袋向量，然后使用训练好的模型进行分类预测。
    
    Args:
        text: 待分类的输入文本
        model: 训练好的分类模型
        char_to_index: 字符到索引的映射字典
        vocab_size: 词汇表的大小
        max_len: 文本的最大长度
        index_to_label: 从索引到标签的映射字典
        
    Returns:
        预测的文本类别标签
    """
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


# 创建索引到标签的反向映射字典
index_to_label = {i: label for label, i in label_to_index.items()}

# 使用训练好的模型对新文本进行分类预测
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
