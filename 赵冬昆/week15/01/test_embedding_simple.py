
"""Test embedding service (no emoji)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import EmbeddingService

print("=== Testing Embedding Service ===")

try:
    embedding_service = EmbeddingService()
    print("Embedding Service initialized")
    
    # Test embedding
    text = "汽车的基本构造包括哪些部分？"
    print(f"\nTest text: {text}")
    
    embedding = embedding_service.embed(text)
    print(f"Generated vector length: {len(embedding)}")
    if len(embedding) > 0:
        print(f"First 5 values: {embedding[:5]}")
        
except Exception as e:
    print(f"Embedding Service test failed: {e}")
    import traceback
    traceback.print_exc()
