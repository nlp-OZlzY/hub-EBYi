
import sqlite3
import pandas as pd
from typing import Union
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData


class ChinookNL2SQLAgent:
    """
    Chinook数据库自然语言到SQL转换Agent
    """

    def __init__(self, db_path: str = './chinook.db'):
        """
        初始化数据库连接

        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        print(f"✅ 成功连接到数据库 chinook.db")
        print(f"📊 发现 {len(self.table_names)} 张表:")
        for i, table in enumerate(self.table_names, 1):
            count = self._get_table_count(table)
            print(f"   {i}. {table} ({count} 条记录)")

    def _get_table_count(self, table_name: str) -> int:
        """
        获取指定表中的记录数量
        """
        table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
        query = select(func.count()).select_from(table_instance)
        result = self.conn.execute(query).fetchone()[0]  # type: ignore[index]
        return result

    def process_question(self, question: str) -> str:
        """
        处理用户的自然语言问题

        Args:
            question: 用户的自然语言问题

        Returns:
            问题的答案
        """
        question_lower = question.lower().strip()

        # 问题1: 数据库中总共有多少张表
        if any(keyword in question_lower for keyword in
               ["多少张表", "表数量", "表格数量", "表的总数", "一共有几张表", "表个数", "total tables",
                "number of tables"]):
            table_count = len(self.table_names)
            return f"数据库中总共有 {table_count} 张表。"

        # 问题2: 员工表中有多少条记录
        elif any(keyword in question_lower for keyword in
                 ["员工表", "员工数量", "employee", "员工记录数", "员工有多少", "employees", "how many employees"]):
            if 'employees' not in self.table_names:
                return "数据库中没有找到employees表"
            count = self._get_table_count('employees')
            return f"员工表中有 {count} 条记录。"

        # 问题3: 客户个数和员工个数分别是多少
        elif any(keyword in question_lower for keyword in
                 ["客户个数", "员工个数", "客户和员工", "客户数", "员工数", "customers and employees",
                  "customer and employee count"]):
            if 'customers' not in self.table_names:
                return "数据库中没有找到customers表"
            if 'employees' not in self.table_names:
                return "数据库中没有找到employees表"

            customer_count = self._get_table_count('customers')
            employee_count = self._get_table_count('employees')
            return f"数据库中客户个数为 {customer_count}，员工个数为 {employee_count}。"

        else:
            return "我暂时无法回答此类问题"

    def close_connection(self):
        """
        关闭数据库连接
        """
        self.conn.close()


def main():
    """
    主函数
    """

    try:
        # 创建Agent实例
        agent = ChinookNL2SQLAgent()

        demo_questions = [
            "数据库中总共有多少张表",
            "员工表中有多少条记录",
            "在数据库中所有客户个数和员工个数分别是多少"
        ]

        for i, q in enumerate(demo_questions, 1):
            print(f"\n{i}. 问题: {q}")
            answer = agent.process_question(q)
            print(f"   回答: {answer}")

    except FileNotFoundError:
        print("❌ 错误: 找不到 chinook.db 文件")
        print("请确保 chinook.db 数据库文件位于当前目录中")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        try:
            agent.close_connection()
        except:
            pass


if __name__ == "__main__":
    main()