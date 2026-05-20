from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ============== 配置 ==============
MODEL_CONFIG = {
    "model": "qwen-max",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-54703c491c0c42bb9dddbc6db21b78da"
}

EMBEDDING_CONFIG = {
    "model_name": "D:\\Code\\STUDY\\models\\BAAI\\bge-small-zh-v1.5",
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": True}
}

ES_CONFIG = {
    "es_url": "http://localhost:9200",
    "index_name": "knowledge_base",
    "verify_certs": False
}

# ============== 文档处理 ==============
def load_documents(folder_path: str):
    """加载文件夹中的所有文档"""
    import os
    from langchain_core.documents import Document

    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # 尝试 utf-8，失败则用 gbk 或 utf-16
                for enc in ("utf-8", "gbk", "utf-16", "utf-16-le"):
                    try:
                        with open(file_path, "r", encoding=enc) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                docs.append(Document(page_content=content, metadata={"source": file_path}))
    return docs

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """根据文档大小自动设置分块参数"""
    total_chars = sum(len(doc.page_content) for doc in documents)

    if total_chars < 1000:
        chunk_size = 200
        chunk_overlap = 40
    elif total_chars < 5000:
        chunk_size = 400
        chunk_overlap = 80
    elif total_chars < 20000:
        chunk_size = 600
        chunk_overlap = 120
    else:
        chunk_size = 800
        chunk_overlap = 160

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ============== 混合检索 ==============
class HybridSearch:
    """BM25 + 向量混合检索"""

    def __init__(self, es_url: str, index_name: str, embeddings):
        self.es_url = es_url
        self.index_name = index_name
        self.embeddings = embeddings
        self.vectorstore = None

    def create_index(self, documents):
        """创建ES索引并存储文档"""
        self.vectorstore = ElasticsearchStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            es_url=self.es_url,
            verify_certs=ES_CONFIG["verify_certs"]
        )

    def similarity_search_with_score(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        混合检索：alpha控制BM25和向量检索的权重
        alpha=0.0 只用BM25，alpha=1.0 只用向量检索
        """
        from elasticsearch import Elasticsearch
        es = Elasticsearch(ES_CONFIG["es_url"])

        # 获取索引的embedding字段名
        mapping = es.indices.get_mapping(index=self.index_name)
        text_field = "text"
        vector_field = "vector"

        # 准备查询
        query_embedding = self.embeddings.embed_query(query)

        # 1. BM25检索
        bm25_body = {
            "size": k * 2,
            "query": {
                "match": {
                    text_field: query
                }
            }
        }
        bm25_results = es.search(index=self.index_name, body=bm25_body)

        # 2. 向量检索 - 兼容不同ES版本
        vector_body = {
            "size": k * 2,
            "query": {
                "knn": {
                    "field": vector_field,
                    "query_vector": query_embedding,
                    "num_candidates": k * 2
                }
            }
        }
        vector_results = es.search(index=self.index_name, body=vector_body)

        # 3. RRF融合（Reciprocal Rank Fusion）
        def rrf_score(rank, k=60):
            return 1 / (k + rank)

        bm25_scores = {}
        for i, hit in enumerate(bm25_results["hits"]["hits"]):
            doc_id = hit["_id"]
            bm25_scores[doc_id] = rrf_score(i) * (1 - alpha) * hit["_score"]

        vector_scores = {}
        for i, hit in enumerate(vector_results["hits"]["hits"]):
            doc_id = hit["_id"]
            vector_scores[doc_id] = rrf_score(i) * alpha * hit["_score"]

        # 合并分数
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined = {}
        for doc_id in all_doc_ids:
            combined[doc_id] = bm25_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)

        # 获取top k
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        # 获取文档内容
        doc_body = {
            "size": k,
            "query": {
                "ids": {
                    "values": [doc_id for doc_id, _ in sorted_docs]
                }
            }
        }
        doc_results = es.search(index=self.index_name, body=doc_body)

        docs_with_scores = []
        for hit in doc_results["hits"]["hits"]:
            docs_with_scores.append((hit["_source"], combined[hit["_id"]]))
        return docs_with_scores

# ============== RAG 链 ==============
def create_rag_chain(hybrid_search, llm):
    """创建检索增强生成链"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，基于提供的上下文回答问题。如果上下文中没有相关信息，请如实说明。"),
        ("human", "上下文：{context}\n\n问题：{question}")
    ])

    def format_context(docs_with_scores):
        context_parts = []
        for doc, score in docs_with_scores:
            context_parts.append(doc.get("text", str(doc)))
        return "\n\n".join(context_parts)

    def retrieve(question):
        results = hybrid_search.similarity_search_with_score(question, k=3, alpha=0.5)
        return format_context(results)

    chain = (
        {"context": retrieve, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ============== 问答接口 ==============
class KnowledgeBaseQA:
    def __init__(self, docs_folder: str):
        print("正在加载文档...")
        docs = load_documents(docs_folder)
        print(f"加载了 {len(docs)} 个文档")

        print("正在分割文档...")
        chunks = split_documents(docs)
        print(f"分割为 {len(chunks)} 个块")

        print("正在初始化嵌入模型...")
        embeddings = HuggingFaceBgeEmbeddings(**EMBEDDING_CONFIG)

        print("正在创建Elasticsearch索引...")
        hybrid_search = HybridSearch(
            es_url=ES_CONFIG["es_url"],
            index_name=ES_CONFIG["index_name"],
            embeddings=embeddings
        )
        hybrid_search.create_index(chunks)
        self.hybrid_search = hybrid_search

        print("正在构建问答链...")
        llm = ChatOpenAI(**MODEL_CONFIG)
        self.chain = create_rag_chain(hybrid_search, llm)
        print("问答系统就绪")

    def ask(self, question: str) -> str:
        """问答"""
        return self.chain.invoke(question)

if __name__ == "__main__":
    qa = KnowledgeBaseQA("D:/code/study/task/week14/1/knowledge_base")

    while True:
        question = input("\n请输入问题（输入 q 退出）：")
        if question.lower() == "q":
            break
        answer = qa.ask(question)
        print(f"\n答案：{answer}")
