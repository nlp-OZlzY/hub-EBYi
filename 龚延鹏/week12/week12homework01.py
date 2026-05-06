import sqlite3

# 连接到Chinook数据库
conn = sqlite3.connect(r'D:\BaiduNetdiskDownload\第12周-ChatBI数据智能问答\第12周-ChatBI数据智能问答\Week12\04_SQL-Code-Agent-Demo\chinook.db')

# 创建一个游标对象
cursor = conn.cursor()

# 获取数据库中所有表的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(tables)

cursor.execute("SELECT * FROM employees ;")
a = cursor.fetchall()
print(a)

cursor.execute("SELECT count(distinct name) FROM sqlite_master WHERE type='table';")
b = cursor.fetchall()
print(b)

cursor.execute("SELECT * FROM customers ;")
c = cursor.fetchall()
print(c)


import time
import jwt
import requests
from itertools import combinations
import numpy as np
from tqdm import tqdm

'''数据库解析'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData
import pandas as pd

class DBParser:
    '''DBParser'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                # 获取当前字段
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计most frequency value
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 任意取值
                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            self.engine.execute(sql)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.engine.execute(sql)
        return list(result)

    def get_full_schema(self) -> str:
        '''获取所有表的简要Schema信息，用于Agent理解全局结构'''
        schema_info = f"数据库中共有 {len(self.table_names)} 张表。\n"
        schema_info += "表名列表: " + ", ".join(self.table_names) + "\n\n"

        for table_name in self.table_names:
            fields = self.get_table_fields(table_name)
            # 只取字段名和类型，减少Token消耗
            field_desc = ", ".join([f"{row['name']}({row['type']})" for _, row in fields.iterrows()])
            schema_info += f"表 '{table_name}' 的字段: {field_desc}\n"
        return schema_info

parser = DBParser(r'sqlite:///D:\BaiduNetdiskDownload\第12周-ChatBI数据智能问答\第12周-ChatBI数据智能问答\Week12\04_SQL-Code-Agent-Demo\chinook.db')
parser.get_table_sample("albums")
parser.get_table_fields("genres")
import time
import jwt
import requests
from itertools import combinations
import numpy as np
from tqdm import tqdm

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def ask_glm(question, nretry=5):
    if nretry == 0:
        return None

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f"Bearer sk-aaf17dcb8fa140ccbc455d66b2e80205"
    }
    data = {
        "model": "glm-5",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except:
        return ask_glm(question, nretry-1)

question_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。提问是自然语言，且计数和统计类型的问题，请直接输出具体提问，不需要有其他输出：

表名称：{table_name}

需要提问和统计的字段：{field}

表{table_name}样例如下：
{data_sample_mk}

表{table_name} schema如下：
{data_schema}
'''

answer_prompt = '''你是一个专业的数据库专家，现在需要你结合表{table_name}的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

表名称：{table_name}

数据表样例如下：
{data_sample_mk}

数据表schema如下：
{data_schema}

提问：{question}
'''

question_rewrite_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。现在需要你将的下面的提问，转换为用户提问的风格。请直接输出提问，不需要有其他输出，不要直接提到表明：

原始问题：{question}

查询的表：{table_name}
'''

answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

原始问题：{question}

执行SQL：{sql}

原始结果：{answer}
'''

company_name_rewrite_prompt = '''将下面的公司的中文缩写名称，如剔除公司名称中的地域信息，或剔除公司名中的有限责任公司等信息。不要输出其他内容，不是英文缩写名称。

原始公司名：{company_name}
'''

NL2SQL_PROMPT = """
你是一个专业的 SQLite 数据库专家。请根据以下数据库 Schema 信息，将用户的自然语言问题转换为可执行的 SQL 语句。

数据库 Schema:
{schema}

注意：
1. 只输出 SQL 语句，不要包含 Markdown 格式（如sql）。 
2. 确保表名和字段名与 Schema 中完全一致（区分大小写）。 
3. 如果是计数问题，请使用 COUNT(*)。
用户问题: {question} 
SQL: 
"""
def run_agent(question):
    print(f"\n--- 处理问题: {question} ---")
# 1. 获取全局 Schema
    schema = parser.get_full_schema()
# 2. 构造 Prompt

    prompt = NL2SQL_PROMPT.format(schema=schema, question=question)

# 3. 调用大模型生成 SQL
    try:
        schema = parser.get_full_schema()
        prompt = NL2SQL_PROMPT.format(schema=schema, question=question)
        response = ask_glm(prompt)
        if not response:
            print("错误: 大模型返回为空 (可能是网络问题或 API Key 失效)")
            return None

        if 'choices' not in response:
            print(f"错误: 大模型返回格式异常: {response}")
            return None

        sql = response['choices'][0]['message']['content'].strip()
    # 清理可能存在的 Markdown 标记
        sql = sql.replace('', '').strip().rstrip(';')
        print(f"生成的 SQL: {sql}")
    # 4. 执行 SQL
        result = parser.execute_sql(sql)
        print(f"查询结果: {result}")
        return result
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

questions = [
    "数据库中总共有多少张表",
    "员工表中有多少条记录",
    "在数据库中所有客户个数和员工个数分别是多少"
]

for q in questions:
    run_agent(q)