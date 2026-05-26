
"""Test complete flow"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService
from services.embedding_service import EmbeddingService

print("Step 1: Initialize services")
milvus = MilvusService()
embed = EmbeddingService()
client = milvus.client

print("\nStep 2: Drop existing collection")
if 'documents' in client.list_collections():
    client.drop_collection('documents')
    print("Dropped documents collection")

print("\nStep 3: Create collection")
from pymilvus import DataType
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("document_id", DataType.VARCHAR, max_length=64)
schema.add_field("page", DataType.INT64)
schema.add_field("metadata", DataType.JSON)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1152)
client.create_collection('documents', schema=schema)
print("Created documents collection")

print("\nStep 4: Insert test data")
test_data = [
    {"text": "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。", "page": 1},
    {"text": "汽车保养周期建议每5000公里进行一次。", "page": 5},
    {"text": "领克客服热线是4006-010101。", "page": 10}
]

for item in test_data:
    vector = embed.embed(item['text'])
    client.insert(
        collection_name='documents',
        data=[{
            "text": item['text'],
            "document_id": "test",
            "page": item['page'],
            "metadata": {"file_name": "test.pdf"},
            "vector": vector
        }]
    )
    print(f"Inserted: {item['text'][:30]}...")

client.flush(['documents'])
print("Flushed data")

print("\nStep 5: Create index")
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_index('documents', index_params)
print("Created index")

print("\nStep 6: Load collection")
client.load_collection('documents')
time.sleep(2)

print("\nStep 7: Test search")
query_vector = embed.embed("发动机")
results = client.search(
    collection_name='documents',
    data=[query_vector],
    limit=2,
    search_params={"metric_type": "COSINE"},
    output_fields=["text", "page"]
)

print(f"Search results: {len(results[0])}")
for hit in results[0]:
    print(f"\nDistance: {hit.get('distance')}")
    print(f"Entity: {hit.get('entity')}")
