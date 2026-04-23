import os
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 加载环境变量
load_dotenv()

# 1. 连接 Chinook 数据库
db = SQLDatabase.from_uri("sqlite:///chinook.db")

# 2. 初始化大模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,  # 0 保证SQL生成准确
    max_tokens=1024
)

# 3. 创建 SQL Agent（核心：自然语言转SQL + 执行 + 回答）
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,  # 打印执行过程，方便调试
    agent_type="openai-tools"
)

# 4. 定义测试问题
questions = [
    "数据库中总共有多少张表",
    "员工表中有多少条记录",
    "在数据库中所有客户个数和员工个数分别是多少"
]

# 5. 执行问答并输出结果
print("===== Chinook NL2SQL 问答结果 =====")
for i, question in enumerate(questions, 1):
    print(f"\n【提问{i}】{question}")
    try:
        answer = agent.invoke(question)
        print(f"【回答】{answer['output']}")
    except Exception as e:
        print(f"执行出错：{str(e)}")
