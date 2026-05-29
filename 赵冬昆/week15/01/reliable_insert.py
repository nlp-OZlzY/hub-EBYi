
"""Reliable data insertion"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService
from services.embedding_service import EmbeddingService

milvus = MilvusService()
embed = EmbeddingService()
client = milvus.client

# Drop existing
if 'documents' in client.list_collections():
    client.drop_collection('documents')
    print("Dropped existing documents collection")

# Create collection manually
from pymilvus import DataType
print("\nCreating documents collection...")
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("document_id", DataType.VARCHAR, max_length=64)
schema.add_field("page", DataType.INT64)
schema.add_field("metadata", DataType.JSON)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1152)
client.create_collection('documents', schema=schema)
print("Created collection")

# Create index
print("\nCreating index...")
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_index('documents', index_params)
print("Created index")

# Insert data
test_data = [
    {"text": "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。", "page": 1},
    {"text": "汽车保养周期建议每5000公里进行一次。", "page": 5},
    {"text": "领克客服热线是4006-010101。", "page": 10}
]

print("\nInserting data...")
for item in test_data:
    vector = embed.embed(item['text'])
    
    # Direct insert
    result = client.insert(
        collection_name='documents',
        data=[{
            "text": item['text'],
            "document_id": "test_doc",
            "page": item['page'],
            "metadata": {"file_name": "test.pdf"},
            "vector": vector
        }]
    )
    print(f"Inserted page {item['page']}, result: {result}")

# Flush and wait
print("\nFlushing data...")
client.flush(['documents'])
time.sleep(3)

# Verify
client.load_collection('documents')
stats = client.get_collection_stats('documents')
print(f"\nStats after flush: {stats}")

# Query
results = client.query(
    collection_name='documents',
    filter="",
    limit=3,
    output_fields=["text", "page"]
)
print(f"Query found {len(results)} entities")
for r in results:
    print(f"  Page {r.get('page')}: {r.get('text')}")

# Test search
print("\nTesting search...")
query_vector = embed.embed("发动机")
search_results = client.search(
    collection_name='documents',
    data=[query_vector],
    limit=2,
    search_params={"metric_type": "COSINE"},
    output_fields=["text", "page"]
)
print(f"Search found {len(search_results[0])} results")
for hit in search_results[0]:
    print(f"  Distance: {hit.get('distance')}")
    print(f"  Text: {hit.get('entity', {}).get('text', '')}")

print("\nDone!")
