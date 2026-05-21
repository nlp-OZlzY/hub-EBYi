"""
本地知识库问答系统 - 演示版（使用假Embeddings）
适合演示和测试，不需要真实的Embedding API
"""

# 解决 OpenMP 库冲突问题（必须在导入其他库之前设置）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_classic.chains import RetrievalQA

# 配置LLM
llm = ChatOpenAI(
    model="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

# 使用假的Embeddings（用于演示，不调用API）
# 注意：生产环境中应该使用真实的 embeddings
embeddings = FakeEmbeddings(size=1536)

# 主流程
def build_knowledge_base(docs_path):
    """构建知识库"""
    print("📂 正在加载文档...")
    # 1. 加载文档
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    print(f"✅ 加载了 {len(documents)} 个文档")

    # 2. 分割文档
    print("✂️  正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " "]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 分割为 {len(splits)} 个文本块")

    # 3. 创建向量数据库
    print("🔄 正在创建向量数据库（使用假Embeddings）...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ 向量数据库创建完成!\n")

    return vectorstore

def ask(vectorstore, question):
    """问答"""
    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 执行查询
    print(f"❓ 问题: {question}")
    try:
        result = qa_chain.invoke({"query": question})
        print(f"💡 答案: {result['result']}")
        print(f"📚 参考了 {len(result['source_documents'])} 个文档片段")

        # 显示参考文档
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n📄 文档{i}: {doc.metadata.get('source', '未知')}")
            print(f"   内容: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ 错误: {str(e)}")

    print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    try:
        # 构建知识库
        print("="*60)
        print("🎉 知识库问答系统启动中...")
        print("⚠️  注意：此版本使用假Embeddings进行演示")
        print("="*60 + "\n")

        vectorstore = build_knowledge_base("./knowledge_base")

        # 问答示例
        questions = [
            "LangChain 是什么？",
            "RAG 技术的工作原理是什么？",
            "FAISS 有什么优势？"
        ]

        for question in questions:
            ask(vectorstore, question)

    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()
