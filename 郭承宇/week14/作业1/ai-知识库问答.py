import os
import requests

# 文档加载
from langchain_community.document_loaders import TextLoader
# 文本切片
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量存储
from langchain_community.vectorstores import FAISS
# Embedding 基类
from langchain_core.embeddings import Embeddings
# LLM模型
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope 原生 Embedding API 封装"""

    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.model = model
        self._url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 10  # DashScope 原生 API 单次最多 10 条
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                self._url,
                json={
                    "model": self.model,
                    "input": {"texts": batch},
                    "parameters": {"text_type": "document"},
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.extend(
                item["embedding"] for item in data["output"]["embeddings"]
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = requests.post(
            self._url,
            json={
                "model": self.model,
                "input": {"texts": [text]},
                "parameters": {"text_type": "query"},
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["output"]["embeddings"][0]["embedding"]


def load_documents(folder_path):
    all_docs = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.md'):
                filepath = os.path.join(dirpath, filename)
                loader = TextLoader(filepath, encoding='utf-8')
                docs = loader.load()
                all_docs.extend(docs)
    return all_docs


documents = load_documents('knowledge_base')
print(f'共加载 {len(documents)} 个文档片段')

text_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_split.split_documents(documents)
# 过滤掉空白、非字符串、以及太短的片段（如单独的 $$）
chunks = [c for c in chunks if c.page_content and isinstance(c.page_content, str) and len(c.page_content.strip()) > 5]
print(f'共生成 {len(chunks)} 个有效片段')

embeddings = DashScopeEmbeddings(
    api_key='sk-7fc1ac30aa1e4bf5870f19dc707bc224'
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """你是一个人工智能课程助教。请根据以下课程讲义内容回答学生的问题。
如果讲义中没有相关信息，请诚实地说"讲义中未涉及此内容"。

课程讲义参考：
{context}

学生问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    model="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-7fc1ac30aa1e4bf5870f19dc707bc224"
)

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

questions = [
    "什么是博弈？",
    "使用通俗易懂的语言帮我解释什么是神经网络",
    "Alpha-Beta 剪枝的核心思想是什么？",
    "今天天气怎么样？",  # 知识库中没有，应诚实回答
]

for q in questions:
    print(f"\n问题：{q}")
    print(f"回答：{rag_chain.invoke(q)}")
    print("-" * 50)