import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

# 1.数据准备
dataset = pd.read_csv('./log.csv', sep=',', header=None)
print(len(set(dataset[1]))) # {'page', 'info', 'api-req', 'record', 'API', 'im', 'click', 'debug', 'api-res', 'user-click', 'tts'}

# 初始化并拟合标签编码器，将文本标签(如 page)转换为数字标签(如0,1,2....)
lbl = LabelEncoder()
lbl.fit(dataset[1].values)
label_mapping = dict(zip(lbl.transform(lbl.classes_), lbl.classes_))


# 将数据按8:2的比例分割为训练集和测试集
# stratify参数确保训练集和测试集中各类别的样本比例与原始数据集保持一致
x_train, x_test, train_label, test_label = train_test_split(
    list(dataset[0].values[:500]),
    lbl.transform(dataset[1].values[:500]),
    test_size=0.2,
    stratify=dataset[1][:500].values
)
print(len(set(train_label)))

# 加载BERT预训练的分词器(Tokenizer)
# 分词器负责将文本转换为模型可识别的输入ID、注意力掩码等
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')

# 对训练集和测试集的文本进行编码
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 2.数据集和数据加载器
# 自定义数据集类，继承自PyTorch的Dataset
# 用于处理编码后的数据和标签，方便后续批量读取
class NewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    # 获取单个样本的方法
    def __getitem__(self, idx):
        # 从编码字典中提取input_ids,attention_mask等，并转换为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    # 返回数据集总样本数的方法
    def __len__(self):
        return len(self.labels)

# 实例化自定义数据集
train_dataset = NewDataset(train_encoding, train_label) # 单个样本读取的数据集
test_dataset = NewDataset(test_encoding, test_label)

# 使用DataLoader创建批量数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 3.模型和优化器
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese',  num_labels=11)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义优化器
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pre_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pre_flat == labels_flat) / len(labels_flat)

# 4.训练和验证函数
# 定义训练函数
def train():
    # 设置模型为训练模式
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    # 遍历训练数据加载器
    for batch in train_loader:
        optim.zero_grad()
        # 将批次数据移动到指定设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 执行前向传播，得到损失和logits
        outputs = model(input_ids, attention_mask=attention_mask,labels=labels) # 自动计算损失
        loss = outputs[0]
        total_train_loss += loss.item()
        # 反向传播计算梯度
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 更新模型参数
        optim.step()
        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))
    # 打印平均训练损失
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

# 定义验证函数
def validation():
    # 设置模型为评估模式
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        # 在验证阶段不计算梯度
        with torch.no_grad():
            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 执行向前传播
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
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")

# 5. 预测新的message
def predict_new_text(new_tests):
    if isinstance(new_tests, str): # 处理单个文本的情况，统一转为列表
        new_tests = [new_tests]
    # 1.对新文本进行编码（和训练时保持一致的参数）
    encoding = tokenizer(
        new_tests,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt' # 返回Pytorch张量
    )
     # 2.将数据移到指定设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # 3.模型预测（评估模式，不计算梯度）
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    # 4.计算概率softmax和预测标签
    probabilities = torch.softmax(logits, dim=1) # 转换为概率分布
    pred_label_ids = torch.argmax(probabilities, dim=1).cpu().numpy() # 获取最大概率的标签ID
    # 5.将数字标签转换回原始文本类别
    pred_labels = [label_mapping[pid] for pid in pred_label_ids]
    # 获取每个预测的置信度
    pred_porbs = [probabilities[i][pid].item() for i, pid in enumerate(pred_label_ids)]
    return pred_labels, pred_porbs

# 6. 主训练循环
for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()

# ------------------------- 7. 预测新内容示例 ---------------------------
print('\n==================== 开始预测新文本 ========================')
single_text = "点击查看课程按钮"
single_label, single_prob = predict_new_text(single_text)
print(f"文本：{single_text}")
print(f"预测类别：{single_label[0]}，置信度：{single_prob[0]:.4f}")

multi_texts = [
    "url：/core/v1/static/resources 响应成功",
    "流式请求结束",
    "useIM_MESSAGE_RECEIVED"
]
multi_labels, multi_probs = predict_new_text(multi_texts)
for text, label, prob in zip(multi_texts, multi_labels, multi_probs):
    print(f"\n文本：{text}")
    print(f"预测类别：{label}，置信度：{prob:.4f}")


