
#!/usr/bin/env python3
"""
RAG 文档问答演示脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

def demo():
    milvus = MilvusService()
    
    test_questions = [
        "汽车发动机是什么？",
        "汽车保养周期是多久？",
        "如何联系领克客服？"
    ]
    
    print("="*60)
    print("           RAG 文档问答演示")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n【问题 {i}】{question}")
        print("-"*60)
        
        results = milvus.search_text('documents', question, limit=3)
        
        if results:
            for j, result in enumerate(results, 1):
                text = result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
                page = result.get('page', 0)
                file_name = result.get('metadata', {}).get('file_name', '未知')
                distance = result.get('distance', 0)
                
                print(f"\n结果 {j}:")
                print(f"  来源: {file_name} 第{page}页")
                print(f"  相似度: {(1 - distance) * 100:.1f}%")
                print(f"  内容: {text}")
        else:
            print("  未找到相关信息")
        
        print("-"*60)
    
    print("\n演示完成！")

if __name__ == "__main__":
    demo()
