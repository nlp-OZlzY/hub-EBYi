from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
# from langgraph.prebuilt import create_react_agent # 旧版，已废弃
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent # 新版入口
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

# ------------------- 构建 Agent -------------------
# 使用标准的 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，请基于检索到的知识库信息回答用户问题。"),
    MessagesPlaceholder(variable_name="messages"),
])
# 使用新版 create_agent，传入 prompt 参数
agent = create_agent(model, tools, prompt=prompt)
# ------------------- 测试 -------------------
if __name__ == "__main__":
    while True:
        q = input("请输入问题：")
        if q == "exit":
            break

        result = agent.invoke({"messages": [{"role": "user", "content": q}]})

        # --- 调试代码开始：打印所有消息以排查问题 ---
        print("\n--- 消息历史调试 ---")
        for i, msg in enumerate(result["messages"]):
            print(f"[{i}] 类型: {msg.type}")
            # 如果是 AIMessage 且包含工具调用，打印工具信息
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"    -> 调用了工具: {msg.tool_calls}")
            else:
                # 打印文本内容的前50个字符
                content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                print(f"    -> 内容: {content_preview}")

        # 获取最终回答
        last_msg = result["messages"][-1]
        if last_msg.content:
            print("\n回答：", last_msg.content)
        else:
            print("\n回答：(模型返回内容为空，请检查上方调试信息，看是否最后一条是工具调用)")
        # --- 调试代码结束 ---