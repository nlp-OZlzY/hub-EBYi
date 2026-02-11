import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np

from datasets import load_dataset

# 1. 加载本地 CSV 数据集（修正：指定 csv 格式）
dataset = load_dataset(
    "csv",
    data_files=r"D:/nlp20/龚延鹏(作业)/week04/week04homework01/laws.csv"
)

# 2. 提取数据集标签（修正：索引 train 子集）
train_dataset = dataset['train']
label = train_dataset['label']

# 3. 构建标签映射关系（优化：简洁高效）
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(label)
label_to_law = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

print("标签映射关系：", label_to_law)
print("-" * 80)

# 4. 加载训练好的模型和分词器
checkpoint_path = './results/checkpoint-832'  # 使用最后一个checkpoint（最佳模型）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# 设备适配（CPU/GPU 自动切换）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. 设置模型为评估模式（推理阶段必须）
model.eval()


def predict_label(text):
    """
    对输入文本进行分类预测

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

    # 输入数据移到对应设备（CPU/GPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 不计算梯度（推理阶段，提升速度，节省内存）
    with torch.no_grad():
        # 获取模型输出
        outputs = model(**inputs)

        # 处理不同的输出格式，提取 logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]

        # 计算概率分布（使用softmax，转换为0-1的概率值）
        probabilities = torch.softmax(logits, dim=-1)

        # 获取预测类别和置信度
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        # 构建所有类别的概率字典
        all_probabilities = {
            label_to_law[i]: probabilities[0][i].item()
            for i in range(len(label_to_law))
        }

    return {
        'text': text,
        'predicted_label': label_to_law[predicted_class],
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probabilities
    }


def main():
    """
    主函数：测试多个样本
    """
    # 测试样本列表（注意：需与训练数据集的任务类型一致，否则预测结果无意义）
    test_samples = [
        "《中华人民共和国固体废物污染环境防治法》第一百一十六条规定，违反本法规定，经中华人民共和国过境转移危险废物的，由海关责令退运该危险废物，处五十万元以上五百万元以下的罚款。",
        "《中华人民共和国海商法》第二百五十三条规定，【保险赔偿的扣减】被保险人未经保险人同意放弃向第三人要求赔偿的权利，或者由于过失致使保险人不能行使追偿权利的，保险人可以相应扣减保险赔偿。",
        "《中华人民共和国海商法》第一百零一条规定，【卸货港】出租人应当在合同约定的卸货港卸货。合同订有承租人选择卸货港条款的，在承租人未按照合同约定及时通知确定的卸货港时，船长可以从约定的选卸港中自行选定一港卸货。承租人未按照合同约定及时通知确定的卸货港，致使出租人遭受损失的，应当负赔偿责任。出租人未按照合同约定，擅自选定港口卸货致使承租人遭受损失的，应当负赔偿责任。",
        "《中华人民共和国大气污染防治法》第一百零四条规定，违反本法规定，有下列行为之一的，由海关责令改正，没收原材料、产品和违法所得，并处货值金额一倍以上三倍以下的罚款；构成走私的，由海关依法予以处罚：（一）进口不符合质量标准的煤炭、石油焦的；（二）进口挥发性有机物含量不符合质量标准或者要求的原材料和产品的；（三）进口不符合标准的机动车船和非道路移动机械用燃料、发动机油、氮氧化物还原剂、燃料和润滑油添加剂以及其他添加剂的。",
        "《中华人民共和国安全生产法》第九十三条规定，生产经营单位的决策机构、主要负责人或者个人经营的投资人不依照本法规定保证安全生产所必需的资金投入，致使生产经营单位不具备安全生产条件的，责令限期改正，提供必需的资金；逾期未改正的，责令生产经营单位停产停业整顿。有前款违法行为，导致发生生产安全事故的，对生产经营单位的主要负责人给予撤职处分，对个人经营的投资人处二万元以上二十万元以下的罚款；构成犯罪的，依照刑法有关规定追究刑事责任。",
        "《中华人民共和国安全生产法》第十二条规定，国务院有关部门按照职责分工负责安全生产强制性国家标准的项目提出、组织起草、征求意见、技术审查。国务院应急管理部门统筹提出安全生产强制性国家标准的立项计划。国务院标准化行政主管部门负责安全生产强制性国家标准的立项、编号、对外通报和授权批准发布工作。国务院标准化行政主管部门、有关部门依据法定职责对安全生产强制性国家标准的实施进行监督检查。",
        "《中华人民共和国安全生产法》第四十六条规定，生产经营单位的安全生产管理人员应当根据本单位的生产经营特点，对安全生产状况进行经常性检查；对检查中发现的安全问题，应当立即处理；不能处理的，应当及时报告本单位有关负责人，有关负责人应当及时处理。检查及处理情况应当如实记录在案。生产经营单位的安全生产管理人员在检查中发现重大事故隐患，依照前款规定向本单位有关负责人报告，有关负责人不及时处理的，安全生产管理人员可以向主管的负有安全生产监督管理职责的部门报告，接到报告的部门应当依法及时处理。",
        "《中华人民共和国大气污染防治法》第十三条规定，制定燃煤、石油焦、生物质燃料、涂料等含挥发性有机物的产品、烟花爆竹以及锅炉等产品的质量标准，应当明确大气环境保护要求。制定燃油质量标准，应当符合国家大气污染物控制要求，并与国家机动车船、非道路移动机械大气污染物排放标准相互衔接，同步实施。前款所称非道路移动机械，是指装配有发动机的移动机械和可运输工业设备。",
        "《中华人民共和国大气污染防治法》第七十七条规定，省、自治区、直辖市人民政府应当划定区域，禁止露天焚烧秸秆、落叶等产生烟尘污染的物质。",
        "《中华人民共和国海商法》第六十四条规定，【赔偿总额的限制】就货物的灭失或者损坏分别向承运人、实际承运人以及他们的受雇人、代理人提出赔偿请求的，赔偿总额不超过本法第五十六条规定的限额。",
    ]

    # 对每个样本进行预测并打印结果
    for i, sample in enumerate(test_samples, 1):
        result = predict_label(sample)

        print(f"【样本 {i}】")
        print(f"输入文本: {result['text']}")
        print(f"预测标签: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
        print()

        # 打印所有类别的概率（修正：变量名 label → emotion）
        print("各类别概率:")
        sorted_probs = sorted(result['all_probabilities'].items(),
                              key=lambda x: x[1],
                              reverse=True)
        for emotion, prob in sorted_probs:
            print(f"  {label:8s}: {prob:.4f}")

        print("-" * 80)
        print()


if __name__ == "__main__":
    main()