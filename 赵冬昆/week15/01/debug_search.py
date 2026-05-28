
"""Debug search functionality"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

print("=== Debug Search ===")
print(f"USE_MOCK_DATA env: {os.getenv('USE_MOCK_DATA', 'true')}")

# Check Milvus connection
milvus_service = MilvusService()
client = milvus_service.client

# List collections
collections = client.list_collections()
print(f"\nMilvus collections: {collections}")

# Check documents collection
if 'documents' in collections:
    stats = client.get_collection_stats('documents')
    print(f"\nDocuments collection stats: {stats}")
    
    # Try a search
    print("\nTesting search...")
    results = milvus_service.search_text('documents', '汽车发动机', limit=3)
    print(f"Search results count: {len(results)}")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Text: {r.get('text', '')[:200]}...")
        print(f"  Page: {r.get('page', 0)}")
        print(f"  Metadata: {r.get('metadata', {})}")
        print(f"  Distance: {r.get('distance', 0)}")
else:
    print("\n' documents' collection not found!")

print("\n=== Debug Complete ===")
