'''
作业1:  基于今天讲解的langchain 的框架，开发对本地知识库进行问答的逻辑，只需要包括文档检索 + llm回答流程(参考项目2)；
作业2: 定义一个skill，包含对股票的可视化功能，对于股票的周波动、日波动绘制在一个图中，并基于大小给出一个买入卖出的最佳时间建议；
    复用已有的skill， 写一个类似功能的skill
'''
from typing import List, Tuple

import numpy as np
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载文档
# 假设你的知识文档在 ./ 文件夹下
loader = DirectoryLoader('./', glob="*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()

print(f"加载了 {len(documents)} 个文档块")
# 2. 分割文本
# chunk_size: 每个片段的大小, chunk_overlap: 片段间的重叠部分（保持上下文连贯）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)
pure_texts = [t.page_content for t in texts]
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('../../models/jinaai/jina-embeddings-v2-base-zh/') #[11 25 24]
model = SentenceTransformer('../../models/BAAI/bge-small-zh-v1.5/')
texts_embeddings = model.encode(pure_texts, normalize_embeddings=True)

def get_bge_relate_texts(question: str = '什么是langchain',model = model):

    question_embeddings = model.encode(question, normalize_embeddings=True)
    # for query_idx, feat in enumerate(question_embeddings):
    score = question_embeddings @ texts_embeddings.T
    max_score_page_idx = score.argsort()[::-1][:5]
    print(max_score_page_idx)
    return [pure_texts[x] for x in max_score_page_idx]

def get_v3_relate_texts(question: str = '什么是langchain',model = model):

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",  # 模型的代号
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="sk-9c6195bf91f7435d88ea4b819073c92c",
        check_embedding_ctx_length=False,
    )
    # print(len(pure_texts),'aa')
    pure_texts_list = [pure_texts[i:i+10] for i in range(0, len(pure_texts), 10)]
    two_vectors = []
    for p in pure_texts_list:
        two_vectors += embeddings.embed_documents(p)
    # print(len(two_vectors),len(two_vectors[0]))
    question_embeddings = embeddings.embed_documents([question])
    # for query_idx, feat in enumerate(question_embeddings):
    question_embeddings = torch.tensor(question_embeddings)
    two_vectors=torch.tensor(two_vectors)
    score = question_embeddings @ two_vectors.T
    # print(score)
    max_score_page_idx = score.numpy()[0].argsort()[::-1][:5]
    print(max_score_page_idx)
    return [pure_texts[x] for x in max_score_page_idx]

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 定义提示词模板（Prompt）
# 这里我们定义了一个简单的翻译任务，包含 source_lang, target_lang, text 三个变量
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个大模型应用开发专家，熟悉各种大模型agen开发的各种框架和技术，请根据相关的参考资料进行回答。"),
    ("human", "相关的参考资料为：{text}，用户的提问是：{question}")
])

# 2. 实例化大模型（Model）
# 这里以 OpenAI 为例，你也可以替换为其他任何兼容的 LLM
model = ChatOpenAI(model="qwen-flash", temperature=0)

# 3. 实例化输出解析器（Output Parser）
# StrOutputParser 负责把模型返回的 AIMessage 对象提取成纯净的字符串
parser = StrOutputParser()

# 4. 使用 LCEL 管道符 | 组装成链（Chain）
# 数据的流向非常直观：输入变量 -> 提示词格式化 -> 模型推理 -> 解析输出
chain = prompt | model | parser


# 5. 调用链（Invoke）
# 只需要传入一个包含所有变量的字典，链就会自动按顺序执行

def fusion(bge, bm25):
    fusion_result = []
    k = 60  # 超参数，实现发现60
    # 多路检索的 结果的合并
    fusion_score = {}  # 每个页面 最终的打分
    for idx, q in enumerate(bge):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)  # 排在后面，得分更低
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(bm25):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)
    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
    sorted_records = [x[0] for x in sorted_dict][:5]
    return sorted_records

def main(question: str):
    bge = get_bge_relate_texts(question)
    # print('111',bge)
    v3 = get_v3_relate_texts(question)
    # print('222',v3)

    relate_text = fusion(bge, v3)
    # print('relate_text', relate_text)

    text_pair = []
    for text in relate_text:
        text_pair.append([question, text])
    relate_text = rerank(text_pair)
    # print('relate_text_rerank', relate_text)
    result = chain.invoke({
        "text": relate_text,
        "question": question,
    })
    print("回答：", result)

rerank_model_path = 'E:\\ai\\weekcode\\models\\BAAI\\bge-reranker-base\\'
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path)
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rerank(text_pair: List[Tuple[str, str]]) -> np.ndarray:
    if rerank_model:
        with torch.no_grad():
            inputs = rerank_tokenizer(
                text_pair, padding=True, truncation=True,
                return_tensors='pt', max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.data.cpu().numpy()

            rerank_idx = np.argsort(scores)[::-1]
            print('rerank_idx', rerank_idx)
            sorted_records = [text_pair[x][1] for x in rerank_idx]
            return sorted_records
    return text_pair


if __name__ == '__main__':
    main('什么是skill')
    pass

