import sqlite3
from fastmcp import FastMCP
mcp = FastMCP(name="SQLite_Server")

@mcp.tool
def execute_sql(sql:str):
    """Execute the given SQL and return the query result."""
    conn = sqlite3.connect("chinook.db")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    except Exception as e:
        print(e)
    finally:
        conn.close() # 关闭数据库链接，释放数据库资源，不能使用corsor.close()

if __name__ == '__main__':
    mcp.run(transport="sse", port=8900)
    # 默认transport是stdio
