from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain.agents import create_agent
import os

# ------------------- 配置 -------------------
os.environ["DASHSCOPE_API_KEY"] = "sk-691d9a0063e1474a8c1006fc504f9d1d"

model = ChatOpenAI(
    model="qwen3.6-flash-2026-04-16",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.2
)

# ------------------- 加载PDF + 切分 -------------------
def load_and_split():
    loader = PyPDFLoader(r"D:\BaiduNetdiskDownload\第1周：课程介绍与大模型基础\推荐书籍\深度学习500问.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    return splitter.split_documents(docs)

split_docs = load_and_split()
print(f"文档切分完成，共 {len(split_docs)} 个文本块")

# ------------------- 向量库 -------------------
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\BaiduNetdiskDownload\models\BAAI\bge-small-zh-v1.5",
    model_kwargs={"device":"cpu"}
)
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# ------------------- 检索工具 -------------------
@tool
def retrieve_from_knowledge_base(query: str) -> str:
    """从本地知识库中检索与问题相关的信息。"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [retrieve_from_knowledge_base]
# ------------------- 构建 Agent（仅用于检索+回答，无多余功能） -------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，请基于检索到的知识库信息回答用户问题。"),
    # ("user", "{input}"),
    # ("placeholder", "{agent_scratchpad}"),
    # MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="messages"),
])
agent = create_react_agent(model, tools, prompt=prompt)

# agent = create_agent(
#     model=model,
#     tools=tools,
#     state_modifier="你是一个AI助手，请基于检索到的知识库信息回答用户问题。"
# )



# ------------------- 测试 -------------------
# if __name__ == "__main__":
#     while True:
#         q = input("请输入问题：")
#         if q == "exit":
#             break
#         result = agent.invoke({"input": q})
#         print("\n回答：", result["output"])
# if __name__ == "__main__":
#     while True:
#         q = input("请输入问题：")
#         if q == "exit":
#             break
#         # 新版输入格式：必须包含 messages 键
#         result = agent.invoke({"messages": [{"role": "user", "content": q}]})
#         # 新版输出格式：从状态字典的 messages 列表中提取最后一条 AI 回复
#         print("\n回答：", result["messages"][-1].content)
if __name__ == "__main__":
    while True:
        q = input("请输入问题：")
        if q == "exit":
            break
        # LangGraph 1.x 的标准调用方式
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        # 从状态中提取最后一条 AI 消息
        print("\n回答：", result["messages"][-1].content)