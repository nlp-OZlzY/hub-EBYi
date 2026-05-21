
"""Initialize Milvus with test data"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()
client = milvus_service.client

cols = client.list_collections()
print(f"Current collections: {cols}")

if 'documents' not in cols:
    print("Creating documents collection...")
    milvus_service._ensure_collection('documents')
    print("Created documents collection")

try:
    results = client.query(
        collection_name='documents',
        filter="",
        limit=1
    )
    
    if len(results) == 0:
        print("Inserting test data...")
        test_data = [
            {"text": "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。", "page": 5, "file_name": "汽车知识手册.pdf"},
            {"text": "汽车底盘包括传动系统、行驶系统、转向系统和制动系统四个部分。", "page": 12, "file_name": "汽车知识手册.pdf"},
            {"text": "汽车电气系统包括蓄电池、发电机、点火系统和照明系统等。", "page": 8, "file_name": "汽车知识手册.pdf"},
            {"text": "汽车保养的关键在于定期更换机油和机油滤清器，建议每5000公里更换一次。", "page": 15, "file_name": "汽车知识手册.pdf"},
            {"text": "刹车片的检查也是重要的保养项目，当刹车片磨损到3mm以下时应及时更换。", "page": 18, "file_name": "汽车知识手册.pdf"},
        ]
        
        for i, item in enumerate(test_data):
            milvus_service.insert_text(
                collection_name='documents',
                text=item['text'],
                document_id='test_doc',
                page=item['page'],
                metadata={'file_name': item['file_name']}
            )
            print(f"Inserted document {i+1}/{len(test_data)}")
        
        print("Test data inserted successfully")
    else:
        print("Documents collection already has data")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Done!")
