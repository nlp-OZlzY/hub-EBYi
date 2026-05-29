
"""Insert data and test search"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService
from services.embedding_service import EmbeddingService

milvus = MilvusService()
embed = EmbeddingService()
client = milvus.client

# Drop and recreate documents collection
if 'documents' in client.list_collections():
    client.drop_collection('documents')
    print("Dropped documents collection")

# Insert test data
test_data = [
    {"text": "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。发动机通过燃烧燃料将化学能转化为机械能。", "page": 1},
    {"text": "汽车保养周期建议每5000公里进行一次常规保养，包括更换机油和机油滤清器。", "page": 5},
    {"text": "领克汽车客服热线是4006-010101，工作时间为周一至周日9:00-21:00。", "page": 10},
    {"text": "汽车底盘包括传动系统、行驶系统、转向系统和制动系统四个主要部分。", "page": 3},
    {"text": "汽车电气系统包括蓄电池、发电机、启动机和各种用电设备。", "page": 8}
]

print(f"Inserting {len(test_data)} documents...")
for item in test_data:
    milvus.insert_text(
        collection_name='documents',
        text=item['text'],
        document_id='test_doc',
        page=item['page'],
        metadata={'file_name': '汽车知识手册.pdf'}
    )
    print(f"  Inserted page {item['page']}")

# Wait for data to sync
time.sleep(2)

# Test search
print("\nTesting search...")
results = milvus.search_text('documents', '发动机', limit=2)
print(f"Search results for '发动机': {len(results)}")
for r in results:
    print(f"\nResult:")
    print(f"  Page: {r.get('page')}")
    print(f"  Text: {r.get('text')}")
    print(f"  Distance: {r.get('distance')}")

print("\nDone! Now you can use chat_cli.py")
