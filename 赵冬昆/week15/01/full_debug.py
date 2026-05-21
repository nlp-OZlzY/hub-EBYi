
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Open log file
log = open("full_debug_log.txt", "w", encoding="utf-8")

def write(msg):
    log.write(msg + "\n")
    print(msg)

write("=== FULL DEBUG ===")

# Step 1: Check PDF parsing
write("\n[1] Checking PDF parsing...")
from services.mineru_service import MinerUService
mineru = MinerUService()
pdf_path = "./uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf"
write(f"PDF path: {pdf_path}")
write(f"File exists: {os.path.exists(pdf_path)}")

result = mineru.parse_document(pdf_path)
write(f"Parse success: {result.get('success')}")

if result.get('success'):
    chunks = result.get('data', {}).get('chunks', [])
    write(f"Number of chunks: {len(chunks)}")
    if chunks:
        write(f"First chunk page: {chunks[0].get('page')}")
        write(f"First chunk content length: {len(chunks[0].get('content', ''))}")

# Step 2: Test Milvus
write("\n[2] Testing Milvus...")
from services.milvus_service import MilvusService
milvus = MilvusService()

# Drop existing
if 'test_full' in milvus.client.list_collections():
    milvus.client.drop_collection('test_full')
    write("Dropped test_full")

# Insert test data
write("\n[3] Inserting test data...")
try:
    milvus.insert_text(
        collection_name='test_full',
        text='汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。',
        document_id='test',
        page=5,
        metadata={'file_name': '汽车手册.pdf'}
    )
    write("Insert successful")
    
    # Wait a bit
    import time
    time.sleep(1)
    
    # Check stats
    stats = milvus.client.get_collection_stats('test_full')
    write(f"Stats after insert: {stats}")
    
    # Query
    results = milvus.client.query(
        collection_name='test_full',
        filter="",
        limit=1,
        output_fields=["text", "page"]
    )
    write(f"Query results: {len(results)}")
    if results:
        write(f"Text: {results[0].get('text')}")
    
except Exception as e:
    write(f"Insert failed: {e}")
    import traceback
    write(f"Traceback: {traceback.format_exc()}")

log.close()
write("\nDebug complete. Log saved to full_debug_log.txt")
