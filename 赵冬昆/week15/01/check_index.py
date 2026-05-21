
"""Check index configuration"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()
client = milvus_service.client

# Check collection info
info = client.get_collection_info('documents')
print("Collection info:", info)

# Check index
indexes = client.list_indexes('documents')
print("\nIndexes:", indexes)

# Get index info
if indexes:
    for idx in indexes:
        idx_info = client.describe_index('documents', idx)
        print(f"\nIndex {idx} info:", idx_info)

# Try to drop and recreate index
print("\nDropping existing index...")
client.drop_index('documents', 'vector')

print("Creating new index...")
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_index('documents', index_params)
print("Index created")

# Load collection
print("\nLoading collection...")
client.load_collection('documents')
print("Collection loaded")

# Test search again
print("\nTesting search after fixing index...")
results = milvus_service.search_text('documents', '汽车发动机', limit=3)
print(f"Search results: {len(results)}")
for r in results:
    print(f"  Text: {r.get('text', '')[:100]}...")
    print(f"  Page: {r.get('page')}")
    print(f"  Metadata: {r.get('metadata')}")
