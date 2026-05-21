
"""Clean up old collections in Milvus"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

print("=== Cleaning up Milvus collections ===")

try:
    milvus_service = MilvusService()
    client = milvus_service.client
    
    collections = client.list_collections()
    print(f"Current collections: {collections}")
    
    # Drop existing collections if they exist
    for coll in ["documents", "document_images"]:
        if coll in collections:
            client.drop_collection(coll)
            print(f"Dropped collection: {coll}")
    
    print("=== Milvus cleanup complete ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
