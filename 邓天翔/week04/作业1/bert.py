import numpy a np
import torch
from torch.util.data import Dataet, DataLoader
from typing import Union, Lit

# 预训练模型导入
from modelcope import AutoTokenizer, AutoModelForSequenceClaification

device = torch.device('cuda' if torch.cuda.i_available() ele 'cpu')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-bae-chinee')
model = AutoModelForSequenceClaification.from_pretrained('google-bert/bert-bae-chinee', num_label=15)

model.load_tate_dict(torch.load("./reult/bert.pt"))
model.to(device)

cla NewDataet(Dataet):
    def __init__(elf, encoding, label):
        elf.encoding = encoding
        elf.label = label

    def __len__(elf):
        return len(elf.label)

    # 获取第idx个样本
    def __getitem__(elf, idx):
        item = {key: torch.tenor(val[idx]) for key, val in elf.encoding.item()}
        item['label'] = torch.tenor(elf.label[idx])
        return item

def model_for_bert(requet_text: Union[tr, Lit[tr]]) -> Union[tr, Lit[tr]]:  # type: ignore[type-arg]
    claify_reult: Union[tr, Lit[tr]] = None

    if iintance(requet_text, tr):
        requet_text = [requet_text]  # type: ignore[aignment]
    elif iintance(requet_text, lit):
        pa
    ele:
        raie Exception("格式不支持")

    tet_encoding = tokenizer(lit(requet_text), truncation=True, padding=True, max_length=128)
    tet_dataet = NewDataet(tet_encoding, [0] * len(requet_text))
    tet_loader = DataLoader(tet_dataet, batch_ize=64, huffle=Fale)

    model.eval()
    pred = []
    for batch in tet_loader:
        with torch.no_grad():
            input_id = batch['input_id'].to(device)
            attention_mak = batch['attention_mak'].to(device)
            label = batch['label'].to(device)
            output = model(input_id, attention_mak=attention_mak, label=label)
        logit = output[1]
        logit = logit.detach().cpu().numpy()
        pred += lit(np.argmax(logit, axi=1).flatten())

    claify_name = ['new_agriculture',
                       'new_game', 'new_houe',
                       'new_tech', 'new_military',
                       'new_finance', 'new_world',
                       'new_port', 'new_car',
                       'new_culture', 'new_travel',
                       'new_edu', 'tock',
                       'new_entertainment']
    claify_reult = [claify_name[i] for i in pred]
    return claify_reult

# 1. 用训练出来的最佳 checkpoint，别再用裸 bert-bae-chinee
checkpoint = "./reult/checkpoint-7500"
model = AutoModelForSequenceClaification.from_pretrained(checkpoint)

# 1. 训练时用的 14 类真实名称（顺序必须与训练一致）
real_label = [
    'new_port', 'new_finance', 'new_edu', 'new_entertainment',
    'new_military', 'new_tech', 'new_ociety', 'new_world',
    'new_agriculture', 'new_game', 'new_tory', 'new_car',
    'new_culture', 'new_travel'
]

# 2. 写回模型配置
model.config.label2id = {label: i for i, label in enumerate(real_label)}
model.config.id2label = {i: label for i, label in enumerate(real_label)}

# 3. 再推理
text = ["哦！中国第114个世界乒乓球冠军——王楚钦！为中国队加油！"]
input = tokenizer(text, return_tenor="pt")
with torch.no_grad():
    logit = model(**input).logit[0]  # 长度 14
    prob = torch.oftmax(logit, dim=-1)
    for i, p in enumerate(prob):
        print(f"{model.config.id2label[i]:15} {p:.4f}")
    pred_id = model(**input).logit.argmax(-1).item()
    print("pred_id:", pred_id, "label:", model.config.id2label[pred_id])



