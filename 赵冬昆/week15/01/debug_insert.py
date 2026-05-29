
"""Debug document insertion"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mineru_service import MinerUService
from services.milvus_service import MilvusService

# Test 1: Check PDF parsing
print("Test 1: PDF Parsing")
mineru = MinerUService()
pdf_path = "./uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf"
print(f"PDF path: {pdf_path}")
print(f"File exists: {os.path.exists(pdf_path)}")

result = mineru.parse_document(pdf_path)
print(f"Parse success: {result.get('success')}")

if result.get('success'):
    chunks = result.get('data', {}).get('chunks', [])
    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk page: {chunks[0].get('page')}")
        print(f"First chunk content (first 200 chars): {chunks[0].get('content', '')[:200]}")

# Test 2: Check Milvus insertion
print("\nTest 2: Milvus Insertion")
milvus = MilvusService()

# Drop and recreate collection
if 'test_debug' in milvus.client.list_collections():
    milvus.client.drop_collection('test_debug')

# Try inserting a simple text
print("Inserting test data...")
try:
    milvus.insert_text(
        collection_name='test_debug',
        text='汽车发动机是汽车的核心部件',
        document_id='test',
        page=1,
        metadata={'file_name': 'test.pdf'}
    )
    print("Insert successful")
    
    # Verify
    result = milvus.client.query(
        collection_name='test_debug',
        filter="",
        limit=1,
        output_fields=["text", "page"]
    )
    print(f"Query result: {len(result)} entities")
    if result:
        print(f"Text: {result[0].get('text')}")
        
except Exception as e:
    print(f"Insert failed: {e}")
    import traceback
    traceback.print_exc()
