"""
基于LangChain的本地知识库问答系统
功能：文档检索 + LLM回答流程
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============ 配置 ============
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-4fedee4ece6541d3b17a7173f0b3c16f"
LLM_MODEL = "qwen-flash"

EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_API_KEY = "sk-4fedee4ece6541d3b17a7173f0b3c16f"
EMBEDDING_MODEL = "text-embedding-v3"

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


# ============ 1. 文档加载 ============
def load_documents(docs_dir: str):
    """从本地目录加载文档"""
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"文档目录不存在，已创建: {docs_dir}")
        print("请将 .txt 文件放入该目录后重新运行")
        return []

    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    return documents


# ============ 2. 文档分块 ============
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """将文档切分为小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个文本块")
    return chunks


# ============ 3. 向量存储 ============
def create_vector_store(chunks):
    """创建FAISS向量存储"""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=EMBEDDING_BASE_URL,
        openai_api_key=EMBEDDING_API_KEY,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("向量索引创建完成")
    return vector_store


# ============ 4. 构建检索问答链 ============
def build_qa_chain(vector_store):
    """构建检索 + LLM问答链"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_template(
        """你是一个知识库问答助手。请根据以下检索到的参考资料回答用户问题。
如果参考资料中没有相关信息，请如实说明无法从知识库中找到答案。

参考资料：
{context}

问题：{question}

回答："""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever


# ============ 5. 主流程 ============
def main():
    print("=" * 50)
    print("  LangChain 本地知识库问答系统")
    print("=" * 50)

    # 加载文档
    documents = load_documents(DOCS_DIR)
    if not documents:
        return

    # 分块
    chunks = split_documents(documents)

    # 创建向量存储
    vector_store = create_vector_store(chunks)

    # 构建问答链
    qa_chain, retriever = build_qa_chain(vector_store)

    print("\n知识库初始化完成！输入问题进行查询，输入 'quit' 退出\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        # 检索相关文档
        related_docs = retriever.invoke(question)
        print(f"\n检索到 {len(related_docs)} 条相关片段：")
        for i, doc in enumerate(related_docs):
            source = doc.metadata.get("source", "未知")
            print(f"  [{i+1}] 来源: {os.path.basename(source)}")
            print(f"      内容: {doc.page_content[:100]}...")

        # LLM回答
        print("\n回答：")
        answer = qa_chain.invoke(question)
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
