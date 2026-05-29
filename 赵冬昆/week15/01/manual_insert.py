
"""Manual insert test"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus = MilvusService()
client = milvus.client

# List collections
print("Current collections:", client.list_collections())

# Create collection manually
from pymilvus import DataType

print("\nCreating documents collection...")
if 'documents' in client.list_collections():
    client.drop_collection('documents')

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("document_id", DataType.VARCHAR, max_length=64)
schema.add_field("page", DataType.INT64)
schema.add_field("metadata", DataType.JSON)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1152)

client.create_collection('documents', schema=schema)
print("Created documents collection")

# Insert test data
print("\nInserting test data...")
test_texts = [
    "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。发动机通过燃烧燃料产生动力。",
    "汽车保养很重要，建议每5000公里更换一次机油。定期保养可以延长车辆寿命。",
    "领克汽车客服热线是4006-010101，如有任何问题可以随时联系。"
]

from services.embedding_service import EmbeddingService
embed_service = EmbeddingService()

for i, text in enumerate(test_texts):
    embedding = embed_service.embed(text)
    print(f"Inserting text {i+1}: {text[:30]}...")
    
    client.insert(
        collection_name='documents',
        data=[{
            "text": text,
            "document_id": "test_doc",
            "page": i + 1,
            "metadata": {"file_name": "汽车手册.pdf"},
            "vector": embedding
        }]
    )

client.flush(['documents'])
print("\nFlush completed")

# Verify
time.sleep(2)
client.load_collection('documents')
stats = client.get_collection_stats('documents')
print("Stats:", stats)

# Test search
print("\nTesting search...")
results = milvus.search_text('documents', '发动机', limit=2)
print(f"Search results: {len(results)}")
for r in results:
    print(f"Text: {r.get('text')}")
    print(f"Page: {r.get('page')}")
