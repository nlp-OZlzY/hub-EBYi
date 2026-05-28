
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()
client = milvus_service.client

# 1. Check collection exists
print("1. Checking collection...")
cols = client.list_collections()
print(f"Collections: {cols}")

# 2. Load collection
print("\n2. Loading collection...")
try:
    client.load_collection('documents')
    print("Collection loaded successfully")
except Exception as e:
    print(f"Load error: {e}")

# 3. Check if collection is loaded
print("\n3. Checking load status...")
status = client.get_load_state('documents')
print(f"Load status: {status}")

# 4. Query data directly
print("\n4. Querying data...")
results = client.query(
    collection_name='documents',
    filter="",
    limit=2,
    output_fields=["text", "page", "metadata"]
)
print(f"Query results count: {len(results)}")
for r in results:
    print(f"  Text: {r.get('text', '')[:50]}...")

# 5. Test search
print("\n5. Testing search...")
try:
    query_vector = milvus_service._embedding_service.embed("汽车发动机")
    print(f"Query vector length: {len(query_vector)}")
    
    results = client.search(
        collection_name='documents',
        data=[query_vector],
        limit=3,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "page", "metadata"]
    )
    
    print(f"Search results count: {len(results)}")
    if len(results) > 0 and len(results[0]) > 0:
        for hit in results[0]:
            print(f"\n  Hit:")
            print(f"    Distance: {hit.get('distance')}")
            print(f"    Entity: {hit.get('entity')}")
    else:
        print("  No results found")
        
except Exception as e:
    print(f"Search error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
