'''
对本地知识库进行问答，文档检索+llm回答流程
'''
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # 正确位置
from langchain_huggingface import HuggingFaceEmbeddings # 使用本地 Embedding

# 本地pdf
pdf_path = './index.pdf'
    

# 加载pdf
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 切块设置
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=100
)

# 切块
doc_splits = text_splitter.split_documents(documents)

# Add the document chunks to the "vector store" using OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=HuggingFaceEmbeddings(model_name="./BAAI/bge-small-zh-v1.5/", model_kwargs={"local_files_only": True}),
    temperature=1
)

# 检索
retriever = vectorstore.as_retriever(k=6)

# 模型
llm = ChatOpenAI(
    model="qwen-flash", # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-57b72c89134c4c169afba0c426451319",
)


# 提问
query = '这份pdf讲了什么'
docs_data = retriever.invoke(query)
print(docs_data, '查看======================================')
docs_string = "".join(doc.page_content for doc in docs_data)
instructions = f"""你是一个问答助手。请根据以下提供的上下文回答问题。
如果不知道答案，请直接说不知道。尽量简洁。

上下文:
{docs_string}"""

ai_msg = llm.invoke([
    {"role": "system", "content": instructions},
    {"role": "user", "content": query}
])

print("answer:", ai_msg.content)

print("documents:",  docs_string)
