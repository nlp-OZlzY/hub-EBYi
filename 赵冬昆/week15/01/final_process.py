
"""Final document processing script"""
import os
import sys
import uuid
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mineru_service import MinerUService
from services.milvus_service import MilvusService

print("=== Starting Document Processing ===")

# Step 1: Parse PDF
print("\n[1/3] Parsing PDF...")
mineru = MinerUService()
pdf_path = "./uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf"
result = mineru.parse_document(pdf_path)

if not result.get('success'):
    print(f"❌ Parse failed: {result.get('error')}")
    sys.exit(1)

chunks = result.get('data', {}).get('chunks', [])
print(f"✅ Parsed {len(chunks)} chunks")

# Step 2: Insert to Milvus
print("\n[2/3] Inserting to Milvus...")
milvus = MilvusService()

# Drop existing
if 'documents' in milvus.client.list_collections():
    milvus.client.drop_collection('documents')
    print("Dropped existing documents collection")

# Insert
success = 0
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    page = chunk.get('page', 0)
    
    if not content.strip():
        continue
    
    milvus.insert_text(
        collection_name='documents',
        text=content,
        document_id='doc_1',
        page=page,
        metadata={'file_name': '汽车知识手册.pdf'}
    )
    success += 1
    
    if (i + 1) % 100 == 0:
        print(f"Inserted {i+1}/{len(chunks)} chunks")

print(f"✅ Inserted {success} chunks")

# Step 3: Verify
print("\n[3/3] Verifying...")
time.sleep(2)  # Wait for data to sync

if 'documents' in milvus.client.list_collections():
    milvus.client.load_collection('documents')
    
    # Get stats
    stats = milvus.client.get_collection_stats('documents')
    print(f"Collection stats: {stats}")
    
    # Query
    results = milvus.client.query(
        collection_name='documents',
        filter="",
        limit=2,
        output_fields=["text", "page"]
    )
    print(f"Query returned {len(results)} entities")
    
    # Test search
    search_results = milvus.search_text('documents', '汽车', limit=3)
    print(f"\nSearch test: Found {len(search_results)} results")
    for r in search_results:
        print(f"  Page {r.get('page')}: {r.get('text', '')[:60]}...")

print("\n=== Processing Complete ===")
