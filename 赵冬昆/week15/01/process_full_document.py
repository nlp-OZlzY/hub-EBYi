
"""Process full document and insert all pages to Milvus"""
import os
import sys
import uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.orm_model import Base, Document, DocumentStatus
from services.mineru_service import MinerUService
from services.milvus_service import MilvusService

# Output file
output_file = "full_process_log.txt"
log_file = open(output_file, "w", encoding="utf-8")

def log(msg):
    log_file.write(msg + "\n")
    print(msg)

log("=== Starting Full Document Processing ===")

# Connect to database
log("\n[1/5] Connecting to database...")
db_path = "sqlite:///./multimodal_rag.db"
engine = create_engine(db_path, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
log("✓ Database connected")

# Get document
log("\n[2/5] Getting document...")
doc = session.query(Document).filter(Document.id == 2).first()
if not doc:
    log("✗ Document not found")
    sys.exit(1)
log(f"✓ Found document: {doc.file_name}")
log(f"  File path: {doc.file_path}")
log(f"  Current status: {doc.status.value}")

# Parse document
log("\n[3/5] Parsing document...")
mineru_service = MinerUService()
result = mineru_service.parse_document(doc.file_path)

if not result.get('success'):
    log(f"✗ Parse failed: {result.get('error')}")
    sys.exit(1)

chunks = result.get('data', {}).get('chunks', [])
log(f"✓ Parsed {len(chunks)} chunks from {doc.file_name}")

# Insert to Milvus
log("\n[4/5] Inserting to Milvus...")
milvus_service = MilvusService()

# Drop existing documents collection if exists
if 'documents' in milvus_service.client.list_collections():
    milvus_service.client.drop_collection('documents')
    log("  Dropped existing documents collection")

log("  Inserting chunks...")
success_count = 0
fail_count = 0

for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    page = chunk.get('page', 0)
    
    if not content.strip():
        continue
    
    try:
        milvus_service.insert_text(
            collection_name='documents',
            text=content,
            document_id=str(doc.id),
            page=page,
            metadata={
                "file_name": doc.file_name,
                "chunk_id": chunk.get("id", str(uuid.uuid4()))
            }
        )
        success_count += 1
        
        if (i + 1) % 50 == 0:
            log(f"    Inserted {i+1}/{len(chunks)} chunks")
            
    except Exception as e:
        fail_count += 1
        log(f"    Failed to insert chunk {i+1}: {e}")

log(f"✓ Insertion complete: {success_count} successful, {fail_count} failed")

# Update document status
log("\n[5/5] Updating document status...")
doc.status = DocumentStatus.COMPLETED
doc.page_count = len(chunks)
session.commit()
log(f"✓ Document status updated to COMPLETED")

# Verify
log("\n[Verification] Checking Milvus...")
cols = milvus_service.client.list_collections()
log(f"Collections: {cols}")

if 'documents' in cols:
    stats = milvus_service.client.get_collection_stats('documents')
    log(f"Documents stats: {stats}")

log("\n=== Processing Complete ===")
log_file.close()
print(f"\nLog saved to {output_file}")

session.close()
