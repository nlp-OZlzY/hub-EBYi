"""
提问: 数据库中总共有多少张表？
回答: 数据库中共有 11 张表。

提问: 员工表中有多少条记录？
回答: 员工表 "employees" 中共有 8 条记录。

提问: 在数据库中所有客户个数和员工个数分别是多少？
回答: 客户个数为 59，员工个数为 8。
"""


import sqlite3
import re

class ChinookAgent:
    def __init__(self, db_path='chinook.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        # 获取所有真实表名（小写存储，便于匹配）
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        self.table_names = [row[0] for row in self.cursor.fetchall()]

    def execute_sql(self, sql):
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            if not result:
                return '0'
            if len(result) == 1 and len(result[0]) == 1:
                return str(result[0][0])
            return '\n'.join(str(row) for row in result)
        except sqlite3.Error as e:
            return f"错误: {e}"

    def find_table(self, possible_names):
        """根据可能的表名列表（小写）返回数据库中真实存在的表名，找不到返回 None"""
        lower_names = [name.lower() for name in self.table_names]
        for candidate in possible_names:
            if candidate.lower() in lower_names:
                # 返回真实表名（保持原始大小写）
                idx = lower_names.index(candidate.lower())
                return self.table_names[idx]
        return None

    def answer(self, question):
        # 问题1: 多少张表
        if re.search(r'多少张表|表的总数|总共有多少张表', question):
            count = len(self.table_names)
            return f"数据库中共有 {count} 张表。"

        # 问题2: 员工表记录数
        elif re.search(r'员工表.*多少条记录|员工表.*数量', question):
            emp_table = self.find_table(['employees', 'employee', 'Employees'])
            if not emp_table:
                return "数据库中没有员工表（常见的表名为 Employees）。"
            sql = f"SELECT COUNT(*) FROM \"{emp_table}\";"  # 用双引号保留原始大小写
            count = self.execute_sql(sql)
            return f"员工表 \"{emp_table}\" 中共有 {count} 条记录。"

        # 问题3: 客户个数和员工个数
        elif re.search(r'客户个数.*员工个数|所有客户和员工.*数量', question):
            cust_table = self.find_table(['customers', 'customer', 'Customers'])
            emp_table = self.find_table(['employees', 'employee', 'Employees'])
            if not cust_table:
                return "数据库中没有客户表（常见的表名为 Customers）。"
            if not emp_table:
                return "数据库中没有员工表（常见的表名为 Employees）。"
            sql_c = f"SELECT COUNT(*) FROM \"{cust_table}\";"
            sql_e = f"SELECT COUNT(*) FROM \"{emp_table}\";"
            cust_count = self.execute_sql(sql_c)
            emp_count = self.execute_sql(sql_e)
            return f"客户个数为 {cust_count}，员工个数为 {emp_count}。"

        else:
            return "抱歉，我只能回答关于表数量、员工表记录数、客户与员工数量的问题。"

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    agent = ChinookAgent('chinook.db')
    qs = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]
    for q in qs:
        print(f"提问: {q}")
        print(f"回答: {agent.answer(q)}\n")
    agent.close()
