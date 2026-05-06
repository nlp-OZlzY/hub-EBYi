import sqlite3
import os
# 如果你使用 OpenAI 官方库
# from openai import OpenAI
# 如果你使用其他兼容 OpenAI 接口的模型（如 DeepSeek, GLM 等），可以使用下面的方式
import openai


class LLMSQLAgent:
    def __init__(self, db_path, api_key, base_url=None, model_name="gpt-3.5-turbo"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # 初始化 LLM 客户端
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        # 1. 获取数据库 Schema 信息 (参考了你上传的 ipynb 逻辑)
        self.schema_info = self._get_schema_info()
        print(f"✅ Agent 初始化完成。加载模型: {model_name}")

    def _get_schema_info(self):
        """
        读取数据库所有表名和建表语句，作为 Prompt 的上下文
        """
        schema_str = ""
        # 获取所有表名
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        for table in tables:
            table_name = table[0]
            # 获取建表语句 (CREATE TABLE ...)
            self.cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            create_table_sql = self.cursor.fetchone()[0]
            schema_str += f"{create_table_sql};\n\n"

        return schema_str

    def generate_sql(self, question):
        """
        调用 LLM 生成 SQL
        """
        system_prompt = """你是一个专业的 SQL 专家。请根据提供的数据库 Schema (SQLite 语法)，
        将用户的自然语言问题转换为 SQL 查询语句。

        注意：
        1. 只输出 SQL 语句，不要包含 Markdown 格式（如 ```sql），不要包含解释。
        2. 确保 SQL 语法适用于 SQLite。
        3. 如果问题涉及统计数量，请使用 COUNT(*)。
        """

        user_prompt = f"""
        ### 数据库 Schema:
        {self.schema_info}

        ### 用户提问:
        {question}

        ### SQL 查询:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0  # 设置低温度以保证 SQL 生成的确定性
            )
            sql = response.choices[0].message.content.strip()
            # 清理可能存在的 Markdown 标记
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql
        except Exception as e:
            return f"Error calling LLM: {e}"

    def execute_sql(self, sql):
        """
        执行 SQL 并获取结果
        """
        try:
            self.cursor.execute(sql)
            # 获取列名
            columns = [description[0] for description in self.cursor.description]
            # 获取数据
            rows = self.cursor.fetchall()
            return columns, rows
        except Exception as e:
            return None, f"SQL Execution Error: {e}"

    def ask(self, question):
        """
        完整的问答流程
        """
        print(f"\n🤖 用户提问: {question}")

        # 1. 生成 SQL
        print("⏳ 正在生成 SQL...")
        sql = self.generate_sql(question)
        print(f"🔍 生成的 SQL: {sql}")

        # 2. 执行 SQL
        columns, result = self.execute_sql(sql)

        if isinstance(result, str) and "Error" in result:
            return f"❌ 执行出错: {result}"

        # 3. 格式化输出
        if not result:
            return "💡 查询结果为空。"

        response = "✅ 查询结果:\n"
        response += " | ".join(columns) + "\n"
        response += "-" * 20 + "\n"
        for row in result:
            response += " | ".join(map(str, row)) + "\n"

        return response


# ==========================================
# 运行 Agent (请替换为你的 API Key)
# ==========================================
if __name__ == "__main__":
    # 请确保 chinook.db 在当前目录，或者修改路径
    # 替换为你的 LLM API Key (例如 OpenAI, DeepSeek, ZhipuAI 等)
    API_KEY = ""

    # 如果使用非 OpenAI 官方模型，请填写 base_url，例如 "https://api.deepseek.com"
    agent = LLMSQLAgent(
        db_path='chinook.db',
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 如果使用 OpenAI 官方，保持 None
        model_name="qwen3.5-plus"
    )

    # --- 测试你的三个提问 ---

    # 提问1
    print(agent.ask("数据库中总共有多少张表"))

    # 提问2
    print(agent.ask("员工表中有多少条记录"))

    # 提问3
    print(agent.ask("在数据库中所有客户个数和员工个数分别是多少"))