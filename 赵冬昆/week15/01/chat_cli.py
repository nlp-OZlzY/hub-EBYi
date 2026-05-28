
#!/usr/bin/env python3
"""
RAG 文档问答 CLI 工具
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

class ChatCLI:
    def __init__(self):
        self.milvus = MilvusService()
    
    def search(self, query, limit=5):
        results = self.milvus.search_text('documents', query, limit)
        return results
    
    def format_output(self, query, results):
        if not results:
            return "没有找到相关信息。"
        
        output = []
        output.append(f"【问题】{query}")
        output.append("")
        output.append("="*50)
        output.append("【检索结果】")
        
        for i, item in enumerate(results, 1):
            text = item.get('text', '')
            page = item.get('page', 0)
            file_name = item.get('metadata', {}).get('file_name', '未知文件')
            distance = item.get('distance', 0)
            
            output.append("")
            output.append(f"--- 结果 {i} ---")
            output.append(f"来源: {file_name} (第{page}页)")
            output.append(f"相似度: {1 - distance:.2%}")
            output.append("")
            output.append(text)
        
        output.append("")
        output.append(f"共找到 {len(results)} 条相关记录")
        
        return "\n".join(output)
    
    def run(self, query=None):
        if query:
            results = self.search(query)
            output = self.format_output(query, results)
            with open('chat_result.txt', 'w', encoding='utf-8') as f:
                f.write(output)
            print("结果已保存到 chat_result.txt")
        else:
            print("="*50)
            print("     RAG 文档问答系统")
            print("="*50)
            print("输入 'exit' 或 'quit' 退出")
            print("="*50)
            
            while True:
                try:
                    q = input("\n请输入问题: ").strip()
                    if q.lower() in ['exit', 'quit', 'q']:
                        print("再见！")
                        break
                    if not q:
                        continue
                    
                    results = self.search(q)
                    output = self.format_output(q, results)
                    
                    print("\n" + "="*50)
                    print(output)
                    print("="*50)
                    
                    with open('chat_history.txt', 'a', encoding='utf-8') as f:
                        f.write(f"【问题】{q}\n")
                        f.write(f"【时间】{__import__('datetime').datetime.now()}\n")
                        f.write("-"*30 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n再见！")
                    break
                except Exception as e:
                    print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG文档问答CLI")
    parser.add_argument('query', nargs='?', help='查询问题')
    args = parser.parse_args()
    
    cli = ChatCLI()
    cli.run(args.query)

if __name__ == "__main__":
    main()
