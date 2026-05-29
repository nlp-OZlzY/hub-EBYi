
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

print("Step 1: Create MilvusService")
milvus = MilvusService()

print("Step 2: Get client")
client = milvus.client

print("Step 3: List collections")
cols = client.list_collections()
print(f"Collections: {cols}")

print("Step 4: Drop test collection if exists")
if 'test_insert' in cols:
    client.drop_collection('test_insert')

print("Step 5: Insert test data")
milvus.insert_text('test_insert', '汽车发动机是汽车的核心部件', 'test_doc', 1, {'file': 'test.pdf'})
milvus.insert_text('test_insert', '汽车底盘包括传动系统和制动系统', 'test_doc', 2, {'file': 'test.pdf'})
milvus.insert_text('test_insert', '汽车电气系统包括蓄电池和发电机', 'test_doc', 3, {'file': 'test.pdf'})

print("Step 6: Search")
results = milvus.search_text('test_insert', '发动机', limit=2)
print(f"Search results: {len(results)}")
for r in results:
    print(f"Text: {r.get('text')}")
    print(f"Page: {r.get('page')}")
    print(f"Metadata: {r.get('metadata')}")

print("Done!")
