import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np


from datasets import load_dataset

dataset = load_dataset("Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset")

# 从数据集中提取文本和情感标签
emotions = dataset['train']['emotion']

# 使用 LabelEncoder 将中文情感标签转换为数字 (0-7)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(emotions)

label_to_emotion = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

print(label_to_emotion)

# 加载训练好的模型和分词器
checkpoint_path = './results/checkpoint-832'  # 使用最后一个checkpoint（最佳模型）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# 设置模型为评估模式
model.eval()

def predict_emotion(text):
    """
    对输入文本进行情感分类预测
    
    Args:
        text (str): 待预测的中文文本
        
    Returns:
        dict: 包含预测结果和置信度的字典
    """
    # 使用tokenizer对文本进行编码
    inputs = tokenizer(text, 
                     truncation=True, 
                     padding=True, 
                     max_length=64, 
                     return_tensors='pt')
    
    # 不计算梯度（推理阶段）
    with torch.no_grad():
        # 获取模型输出
        outputs = model(**inputs)
        
        # 处理不同的输出格式
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]
        
        # 计算概率分布（使用softmax）
        probabilities = torch.softmax(logits, dim=-1)
        
        # 获取预测类别和置信度
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # 获取所有类别的概率
        all_probabilities = {
            label_to_emotion[i]: probabilities[0][i].item() 
            for i in range(len(label_to_emotion))
        }
    
    return {
        'text': text,
        'predicted_emotion': label_to_emotion[predicted_class],
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probabilities
    }

def main():
    """
    主函数：测试多个样本
    """
    # 测试样本列表
    test_samples = [
        "明天周末有空吗？一起去看电影吧。",
        "太好了！我终于拿到驾照了！",
        "今天下雨了，心情不太好，有点难过。",
        "烦死了！我的手机又坏了，这已经是第三次了！",
        "哇！你居然中了500万彩票，太厉害了吧！",
        "那个人随地吐痰，太恶心了，真没素质！",
        "我和最好的朋友吵架了，心里很难受。",
        "吓死我了！刚才差点被车撞到！",
        "终于考完试了，太爽了，可以好好放松一下！",
        "这家餐厅的菜太难吃了，又贵又难吃，再也不来了。",
    ]
    
    # 对每个样本进行预测
    for i, sample in enumerate(test_samples, 1):
        result = predict_emotion(sample)
        
        print(f"输入文本: {result['text']}")
        print(f"预测情感: {result['predicted_emotion']}")
        print(f"置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print()
        
        # 打印所有情感类别的概率
        print("各情感类别概率:")
        sorted_probs = sorted(result['all_probabilities'].items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        for emotion, prob in sorted_probs:
            print(f"  {emotion:8s}: {prob:.4f}")
        
        print("-" * 80)
        print()

if __name__ == "__main__":
    main()
