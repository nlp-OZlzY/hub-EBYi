
"""Insert all document chunks to Milvus"""
import os
import sys
import uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mineru_service import MinerUService
from services.milvus_service import MilvusService

# Parse PDF
mineru = MinerUService()
pdf_path = "./uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf"
result = mineru.parse_document(pdf_path)

if not result.get('success'):
    print(f"Parse failed: {result.get('error')}")
    sys.exit(1)

chunks = result.get('data', {}).get('chunks', [])
print(f"Parsed {len(chunks)} chunks")

# Insert to Milvus
milvus = MilvusService()

# Drop existing documents collection
if 'documents' in milvus.client.list_collections():
    milvus.client.drop_collection('documents')
    print("Dropped existing documents collection")

# Insert chunks
success = 0
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    page = chunk.get('page', 0)
    
    if not content.strip():
        continue
    
    milvus.insert_text(
        collection_name='documents',
        text=content,
        document_id='full_doc',
        page=page,
        metadata={'file_name': '汽车知识手册.pdf'}
    )
    success += 1
    
    if (i + 1) % 50 == 0:
        print(f"Inserted {i+1}/{len(chunks)} chunks")

print(f"\nDone! Inserted {success} chunks")

# Verify
cols = milvus.client.list_collections()
print(f"Collections: {cols}")

if 'documents' in cols:
    milvus.client.load_collection('documents')
    stats = milvus.client.get_collection_stats('documents')
    print(f"Documents stats: {stats}")
    
    # Test search
    results = milvus.search_text('documents', '汽车', limit=2)
    print(f"\nSearch results for '汽车': {len(results)}")
    for r in results:
        print(f"  Page {r.get('page')}: {r.get('text', '')[:50]}...")
