"""
文档切分完成，共 1293 个文本块
根据检索到的知识库信息，我来为您解答关于NiN的问题：

## NiN是什么？

**NiN（Network in Network）** 是一种卷积神经网络模型，是在AlexNet问世不久后提出的。其主要特点包括：

1. **卷积层设计**：使用卷积窗口形状分别为11×11、5×5和3×3的卷积层，相应的输出通道数与AlexNet中的一致。

2. **池化层设计**：每个NiN块后接一个步幅为2、窗口形状为3×3的最大池化层。

3. **关键创新**：NiN去掉了AlexNet最后的3个全连接层，取而代之的是：
   - 使用输出通道数等于标签类别数的NiN块
   - 然后使用**全局平均池化层**对每个通道中所有元素求平均并直接用于分类

4. **优势**：这种设计可以显著减小模型参数尺寸，从而缓解过拟合问题。不过，该设计有时会造成获得有效模型的训练时间增加。

## 在文档的哪个位置？

NiN相关内容位于文档的 **第5.8.2节 "NiN模型"** 部分。
"""

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-14ddbf6d3e1c41c5ae4e9088e9c6dbfc"

model = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-14ddbf6d3e1c41c5ae4e9088e9c6dbfc",
    temperature=0.2
)


# --- 文档加载与切分 ---
def load_and_split():
    loader = PyPDFLoader("G:/动手学深度学习.pdf")
    docs = loader.load()

    # 使用修正后的 RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    return splitter.split_documents(docs)


split_docs = load_and_split()
print(f"文档切分完成，共 {len(split_docs)} 个文本块")

embeddings = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory="./chroma_db")


# 定义一个检索工具
@tool
def retrieve_from_knowledge_base(query: str) -> str:
    """从本地知识库中检索与问题相关的信息。"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# 创建智能体
agent = create_agent(
    model=model,
    tools=[retrieve_from_knowledge_base],
    system_prompt="你是一个AI助手，请基于检索到的知识库信息回答用户问题。",
)

# 执行问答
result = agent.invoke({"messages": [{"role": "user", "content": "文档中的NiN是什么？在哪个地方？"}]})
print(result["messages"][-1].content)
