import os
import sqlite3
import re
from typing import Union, List, Dict, Any
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import function_tool
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


class DBParser:
    def __init__(self, db_path: str = "chinook.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.table_names = self._get_table_names()
        self.table_schemas = self._get_all_schemas()

    def _get_table_names(self) -> List[str]:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        return [t for t in tables if t != 'sqlite_sequence']

    def _get_table_schema(self, table_name: str) -> Dict[str, Any]:
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample = self.cursor.fetchall()
        return {
            "columns": [{"Name": col[1], "type": col[2]} for col in columns],
            "sample": sample
        }

    def _get_all_schemas(self) -> str:
        schemas = []
        for table_name in self.table_names:
            schema = self._get_table_schema(table_name)
            columns_str = ", ".join([f"{c['Name']} ({c['type']})" for c in schema["columns"]])
            schemas.append(f"表名: {table_name}, 字段: {columns_str}")
        return "\n".join(schemas)

    def check_sql(self, sql: str) -> Union[bool, str]:
        try:
            self.cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def execute_sql(self, sql: str) -> Union[List, None]:
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            return None

    def close(self):
        self.conn.close()


parser = DBParser("chinook.db")


def get_table_structures() -> str:
    return parser.table_schemas


def check_sql(sql: str) -> str:
    is_valid, msg = parser.check_sql(sql)
    if is_valid:
        return f"SQL校验通过: {msg}"
    else:
        return f"SQL校验失败: {msg}"


def run_sql(sql: str) -> str:
    result = parser.execute_sql(sql)
    if result is None:
        return "SQL执行失败"
    return str(result)


@function_tool
def get_all_table_structures() -> str:
    """获取数据库中所有表的结构信息，包括表名和字段。"""
    return get_table_structures()


@function_tool
def validate_sql(sql: str) -> str:
    """校验SQL语句是否正确可执行。使用EXPLAIN QUERY PLAN来验证。"""
    return check_sql(sql)


@function_tool
def execute_sql(sql: str) -> str:
    """执行SQL语句并返回查询结果。"""
    return run_sql(sql)


table_selection_prompt = """你是一个数据库专家，负责根据用户的问题选择需要查询的表。

请分析用户的问题，确定需要查询哪些表。

输出格式：
选择的表：表名1, 表名2
表结构信息：
[粘贴对应的表结构信息]
"""

sql_generation_prompt = """你是一个专业的SQL生成专家，根据用户问题和选择的表结构生成SQL查询语句。

【ReAct推理模式】
1. 理解问题：仔细阅读用户的问题
2. 分析需求：确定需要查询什么数据
3. 编写SQL：根据表结构编写SQL
4. 验证SQL：如果校验失败，根据错误信息重新生成
5. 输出SQL：最终只输出纯SQL语句

【重要】
- 如果SQL校验失败，必须根据错误信息重新生成SQL
- 最终输出必须是纯SQL语句，不要有任何其他文字

用户问题：{question}

选择的表结构：
{selected_tables}

请开始生成SQL。
"""

sql_execution_prompt = """你是一个数据分析专家，负责根据SQL执行结果用自然语言回答用户的问题。

用户原始问题：{question}
生成的SQL：{sql}
执行结果：{result}

请用自然语言总结分析结果，回答用户的问题。
"""


def extract_sql(text: str) -> str:
    patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(SELECT.*?)\s*```',
        r'(SELECT\s+.*?;)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip().strip(';')
    return text.strip().strip(';')


async def run_table_selection_agent(question: str) -> str:
    agent = Agent(
        name="TableSelectionAgent",
        instructions=table_selection_prompt,
        tools=[get_all_table_structures],
        model=OpenAIChatCompletionsModel(
            model="qwen3.5-plus-2026-02-15",
            openai_client=client,
        )
    )

    result = await Runner.run(
        starting_agent=agent,
        input=f"用户问题：{question}\n\n请使用 get_all_table_structures 工具获取表结构，然后选择相关表并输出表结构信息。"
    )

    selected_tables = result.final_output if result.final_output else ""
    print(f"\n{'='*60}")
    print("【Agent 1: 表选择】")
    print('='*60)
    print(selected_tables)
    return selected_tables


async def run_sql_generation_agent(question: str, selected_tables: str) -> str:
    agent = Agent(
        name="SQLGenerationAgent",
        instructions=sql_generation_prompt.format(question=question, selected_tables=selected_tables),
        tools=[validate_sql],
        model=OpenAIChatCompletionsModel(
            model="qwen3.5-plus-2026-02-15",
            openai_client=client,
        )
    )

    max_retries = 3
    sql = ""

    for i in range(max_retries):
        if i == 0:
            result = await Runner.run(
                starting_agent=agent,
                input="请生成SQL并使用 validate_sql 工具验证。如果校验失败，请重新生成。"
            )
        else:
            result = await Runner.run(
                starting_agent=agent,
                input=f"SQL校验失败，请根据错误信息重新生成SQL。这是第{i+1}次尝试。"
            )

        response_text = result.final_output if result.final_output else ""
        print(f"\n{'='*60}")
        print(f"【Agent 2: SQL生成】 (尝试 {i+1})")
        print('='*60)
        print(f"LLM输出:\n{response_text}")

        sql = extract_sql(response_text)
        print(f"\n提取的SQL: {sql}")

        validation_result = check_sql(sql)
        print(f"校验结果: {validation_result}")

        if "校验通过" in validation_result:
            break

    print(f"\n最终SQL: {sql}")
    return sql


async def run_execution_agent(question: str, sql: str) -> str:
    print(f"\n{'='*60}")
    print("【Agent 3: SQL执行与分析】")
    print('='*60)

    result = run_sql(sql)
    print(f"执行结果: {result}")

    agent = Agent(
        name="ExecutionAgent",
        instructions=sql_execution_prompt.format(question=question, sql=sql, result=result),
        model=OpenAIChatCompletionsModel(
            model="qwen3.5-plus-2026-02-15",
            openai_client=client,
        )
    )

    final_result = await Runner.run(
        starting_agent=agent,
        input="请根据执行结果用自然语言回答用户的问题。"
    )

    answer = final_result.final_output if final_result.final_output else str(result)
    print(f"\n最终回答: {answer}")
    return answer


async def main():
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'#'*70}")
        print(f"第 {i} 个问题: {question}")
        print('#'*70)

        selected_tables = await run_table_selection_agent(question)
        sql = await run_sql_generation_agent(question, selected_tables)
        answer = await run_execution_agent(question, sql)

        print(f"\n{'='*70}")
        print(f"问题 {i} 回答: {answer}")
        print('='*70)

    parser.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
