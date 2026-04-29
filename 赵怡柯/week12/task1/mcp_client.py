import os

os.environ["OPENAI_API_KEY"] = "sk-54703c491c0c42bb9dddbc6db21b78da"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def run(mcp_server: MCPServer):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    # 创建第一个 AI Agent
    # 这个Agent默认策略：会自动多轮对话、自动调用工具
    agent = Agent(
        name="Assistant",
        instructions="""
        你是专业的SQL数据分析助手，基于chinook.db数据库回答问题。
        你需要：
        1. 理解用户的自然语言问题
        2. 自动生成正确的SQL语句
        3. 查询表数量时，必须排除 name 以 'sqlite_' 开头的系统表
        4. 通过工具 query_database 查询数据库
        5. 返回清晰、简洁的自然语言答案
        """, # 系统提示词（可写角色设定）
        mcp_servers=[mcp_server], # 绑定外部工具（天气查询）
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,
        )
    )

    message1 = "数据库中总共有多少张表。"
    print(f"Running: {message1}")
    result1 = await Runner.run(starting_agent=agent, input=message1)
    # print(result1.new_items)
    # print("\n\n----\n\n")
    print("最终答案：", result1.final_output)

    print("\n----\n")

    message2 = "员工表中有多少条记录。"
    print(f"Running: {message2}")
    result2 = await Runner.run(starting_agent=agent, input=message2)
    # print(result2.new_items)
    # print("\n\n----\n\n")
    print("最终答案：", result2.final_output)

    print("\n----\n")

    message3 = "在数据库中所有客户个数和员工个数分别是多少。"
    print(f"Running: {message3}")
    result3 = await Runner.run(starting_agent=agent, input=message3)
    # print(result3.new_items)
    # print("\n\n----\n\n")
    print("最终答案：", result3.final_output)

    print("\n----\n")

async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",  # sql工具服务
            },
    )as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(main())

# AgentSDK全程自动完成多轮LLM调用 + 上下文拼接