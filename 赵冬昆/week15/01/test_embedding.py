
"""Test embedding service"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import EmbeddingService

print("=== 测试 Embedding Service ===")

try:
    embedding_service = EmbeddingService()
    print("✅ Embedding Service 初始化成功")
    
    # Test embedding
    text = "汽车的基本构造包括哪些部分？"
    print(f"\n测试文本: {text}")
    
    embedding = embedding_service.embed(text)
    print(f"✅ 生成向量长度: {len(embedding)}")
    if len(embedding) > 0:
        print(f"前5个值: {embedding[:5]}")
        
except Exception as e:
    print(f"❌ Embedding Service 测试失败: {e}")
    import traceback
    traceback.print_exc()
