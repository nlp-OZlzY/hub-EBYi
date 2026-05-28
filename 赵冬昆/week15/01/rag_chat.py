
#!/usr/bin/env python3
"""
RAG Chat CLI - 文档问答命令行工具

用法:
    python rag_chat.py                    # 进入交互式模式
    python rag_chat.py "你的问题"          # 直接提问
    python rag_chat.py --help             # 显示帮助信息
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

class RagChatCLI:
    def __init__(self):
        self.milvus_service = MilvusService()
        self.session_id = None
        
    def search_documents(self, query, limit=5):
        """搜索相关文档"""
        text_results = self.milvus_service.search_text(
            collection_name='documents',
            query=query,
            limit=limit
        )
        return text_results
    
    def format_answer(self, query, results):
        """格式化答案"""
        if not results:
            return "未找到相关信息。", []
        
        context_parts = []
        sources = []
        
        for item in results:
            text = item.get('text', '')
            page = item.get('page', 0)
            metadata = item.get('metadata', {})
            file_name = metadata.get('file_name', 'unknown')
            
            context_parts.append(f"[来源: {file_name} 第{page}页]\n{text}")
            sources.append({
                "file_name": file_name,
                "page": page,
                "content_preview": text[:100] + "..." if len(text) > 100 else text
            })
        
        answer = f"根据检索到的信息，关于 \"{query}\" 的回答如下：\n\n"
        answer += "\n\n---\n\n".join(context_parts)
        answer += f"\n\n（共检索到 {len(results)} 条相关记录）"
        
        return answer, sources
    
    def chat(self, query):
        """处理单个问题"""
        results = self.search_documents(query)
        answer, sources = self.format_answer(query, results)
        return answer, sources
    
    def run_interactive(self):
        """运行交互式模式"""
        print("=" * 60)
        print("          RAG 文档问答 CLI")
        print("=" * 60)
        print("输入问题开始问答，输入 'exit' 或 'quit' 退出")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n请输入问题: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("再见！")
                    break
                
                if not query:
                    continue
                
                print("\n正在检索...")
                answer, sources = self.chat(query)
                
                print("\n" + "=" * 60)
                print("答 案:")
                print("-" * 60)
                print(answer)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n出错了: {e}")
    
    def run_single(self, query):
        """运行单次查询模式"""
        try:
            answer, sources = self.chat(query)
            print(answer)
            
            if sources:
                print("\n来源:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source['file_name']} (第{source['page']}页)")
                    
        except Exception as e:
            print(f"出错了: {e}", file=sys.stderr)
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="RAG Chat CLI - 文档问答命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rag_chat.py                    # 进入交互式对话
  python rag_chat.py "汽车发动机是什么？"  # 直接提问
  python rag_chat.py --help             # 显示帮助
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='要查询的问题（省略则进入交互式模式）'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='返回结果数量限制（默认5条）'
    )
    
    args = parser.parse_args()
    
    cli = RagChatCLI()
    
    if args.query:
        cli.run_single(args.query)
    else:
        cli.run_interactive()

if __name__ == "__main__":
    main()
