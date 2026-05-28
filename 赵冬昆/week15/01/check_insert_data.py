
"""Check inserted data in Milvus"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()
client = milvus_service.client

print("Querying data from documents collection...")
results = client.query(
    collection_name='documents',
    filter="",
    limit=3,
    output_fields=["*"]
)

print(f"Found {len(results)} entities")
for i, r in enumerate(results):
    print(f"\nEntity {i+1}:")
    print(f"  Type: {type(r)}")
    print(f"  Keys: {r.keys()}")
    for key, value in r.items():
        print(f"  {key}: {type(value)} = {value[:100] if isinstance(value, str) else value}")
