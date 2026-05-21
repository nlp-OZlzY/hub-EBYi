
"""Check data in Milvus documents collection"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus = MilvusService()
client = milvus.client

# Check collections
cols = client.list_collections()
print(f"Collections: {cols}")

# Check documents collection
if 'documents' in cols:
    print("\n=== Documents Collection ===")
    
    # Load collection
    client.load_collection('documents')
    
    # Get stats
    stats = client.get_collection_stats('documents')
    print(f"Stats: {stats}")
    
    # Query data
    print("\nQuerying documents...")
    results = client.query(
        collection_name='documents',
        filter="",
        limit=3,
        output_fields=["text", "page", "document_id"]
    )
    print(f"Found {len(results)} entities")
    
    for i, r in enumerate(results):
        print(f"\nEntity {i+1}:")
        print(f"  Page: {r.get('page')}")
        print(f"  Doc ID: {r.get('document_id')}")
        text = r.get('text', '')
        print(f"  Text length: {len(text)}")
        print(f"  Text preview: {text[:100]}...")

# Test search
print("\n=== Testing Search ===")
results = milvus.search_text('documents', '汽车发动机', limit=3)
print(f"Search results: {len(results)}")
for r in results:
    print(f"\nResult:")
    print(f"  Text: {r.get('text', '')[:100]}...")
    print(f"  Page: {r.get('page')}")
    print(f"  Distance: {r.get('distance')}")
