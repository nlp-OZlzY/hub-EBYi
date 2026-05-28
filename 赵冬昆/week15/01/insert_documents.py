
"""Insert documents to Milvus"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.orm_model import Base, Document
from services.mineru_service import MinerUService
from services.milvus_service import MilvusService
import uuid

# Connect to database
db_path = "sqlite:///./multimodal_rag.db"
engine = create_engine(db_path, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Get document
doc = session.query(Document).filter(Document.id == 2).first()
if not doc:
    print("Document not found")
    sys.exit(1)

# Parse document
mineru_service = MinerUService()
result = mineru_service.parse_document(doc.file_path)

if not result.get('success'):
    print(f"Parse failed: {result.get('error')}")
    sys.exit(1)

chunks = result.get('data', {}).get('chunks', [])
print(f"Parsed {len(chunks)} chunks")

# Insert to Milvus
milvus_service = MilvusService()

# Drop existing documents collection if exists
if 'documents' in milvus_service.client.list_collections():
    milvus_service.client.drop_collection('documents')
    print("Dropped existing documents collection")

# Insert chunks
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    page = chunk.get('page', 0)
    
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
    
    if (i + 1) % 50 == 0:
        print(f"Inserted {i+1}/{len(chunks)} chunks")

print(f"Successfully inserted {len(chunks)} chunks to 'documents' collection")

# Verify insertion
cols = milvus_service.client.list_collections()
print(f"Current collections: {cols}")

if 'documents' in cols:
    stats = milvus_service.client.get_collection_stats('documents')
    print(f"Documents stats: {stats}")

session.close()
print("Done!")
