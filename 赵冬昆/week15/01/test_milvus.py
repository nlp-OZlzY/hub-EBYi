
"""Test Milvus functionality"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

print("=== Testing Milvus ===")

milvus_service = MilvusService()

# Insert some test data
print("\nInserting test data...")
test_texts = [
    "汽车发动机是汽车的核心部件，负责产生动力。",
    "汽车底盘包括传动系统、行驶系统、转向系统和制动系统。",
    "汽车电气系统包括蓄电池、发电机、点火系统等。",
    "汽车车身是汽车的外壳，保护内部部件和乘客。"
]

for i, text in enumerate(test_texts):
    milvus_service.insert_text(
        collection_name="test_docs",
        text=text,
        document_id="test_doc_1",
        page=i + 1,
        metadata={"test": True}
    )
    print(f"Inserted text {i+1}")

# Test search
print("\nSearching for '发动机'...")
results = milvus_service.search_text("test_docs", "发动机", limit=3)
print(f"Found {len(results)} results")
for i, r in enumerate(results):
    print(f"  Result {i+1}: {r.get('text', '')}")

print("\nSearching for '底盘'...")
results = milvus_service.search_text("test_docs", "底盘", limit=3)
print(f"Found {len(results)} results")
for i, r in enumerate(results):
    print(f"  Result {i+1}: {r.get('text', '')}")

print("\n=== Milvus test complete ===")
