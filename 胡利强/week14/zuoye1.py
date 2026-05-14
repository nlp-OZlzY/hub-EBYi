from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# ---------------------- 1. 加载本地文档 ----------------------
# 你只需要把本地知识库改成自己的 .txt 文件路径
loader = TextLoader("local_knowledge.txt", encoding="utf-8")
documents = loader.load()

# ---------------------- 2. 文档分块 ----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)

# ---------------------- 3. 构建向量数据库 ----------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(split_docs, embeddings)

# ---------------------- 4. 检索器 + LLM 问答链 ----------------------
# 使用本地免费模型（无需API Key）
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    model_kwargs={"max_length": 512}
)

# 检索 + 回答 核心链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# ---------------------- 5. 测试问答 ----------------------
if __name__ == "__main__":
    query = input("请输入你的问题：")
    result = qa_chain({"query": query})
    
    print("\n【AI回答】")
    print(result["result"])
    
    print("\n【参考来源】")
    for idx, doc in enumerate(result["source_documents"]):
        print(f"来源{idx+1}：{doc.page_content[:100]}...")
