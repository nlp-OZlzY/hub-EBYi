
#!/usr/bin/env python3
"""
RAG Chat CLI - 文档问答命令行工具
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

def search_documents(query, limit=5):
    milvus_service = MilvusService()
    results = milvus_service.search_text(
        collection_name='documents',
        query=query,
        limit=limit
    )
    return results

def format_answer(query, results):
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
            "page": page
        })
    
    answer = f"问题: {query}\n\n"
    answer += "=" * 60 + "\n"
    answer += "\n\n---\n\n".join(context_parts)
    answer += f"\n\n（共检索到 {len(results)} 条相关记录）"
    
    return answer, sources

def main():
    parser = argparse.ArgumentParser(description="RAG Chat CLI")
    parser.add_argument('query', nargs='?', help='要查询的问题')
    parser.add_argument('--limit', type=int, default=5, help='返回结果数量')
    args = parser.parse_args()
    
    if not args.query:
        print("请提供查询问题，如: python rag_cli.py \"你的问题\"", file=sys.stderr)
        sys.exit(1)
    
    results = search_documents(args.query, args.limit)
    answer, sources = format_answer(args.query, results)
    
    with open('rag_result.txt', 'w', encoding='utf-8') as f:
        f.write(answer)
        if sources:
            f.write("\n\n来源列表:\n")
            for i, source in enumerate(sources, 1):
                f.write(f"  {i}. {source['file_name']} (第{source['page']}页)\n")
    
    print("结果已保存到 rag_result.txt")

if __name__ == "__main__":
    main()
