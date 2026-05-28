
"""Test document processing step by step"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.orm_model import Base, Document, DocumentStatus
from services.mineru_service import MinerUService
from services.milvus_service import MilvusService

print("=== Step 1: Check database ===")
db_path = "sqlite:///./multimodal_rag.db"
engine = create_engine(db_path, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

doc = session.query(Document).filter(Document.id == 2).first()
print(f"Document 2 found: {doc.file_name if doc else 'Not found'}")
if doc:
    print(f"Status: {doc.status.value}")
    print(f"File path: {doc.file_path}")

print("\n=== Step 2: Test MinerU parsing ===")
mineru_service = MinerUService()
if doc and os.path.exists(doc.file_path):
    result = mineru_service.parse_document(doc.file_path)
    print(f"Parse success: {result.get('success')}")
    if result.get('success'):
        chunks = result.get('data', {}).get('chunks', [])
        images = result.get('data', {}).get('images', [])
        print(f"Chunks: {len(chunks)}, Images: {len(images)}")
        if chunks:
            print(f"First chunk (preview): {chunks[0].get('content', '')[:200]}")
    else:
        print(f"Error: {result.get('error')}")
else:
    print("Document or file not found")

print("\n=== Step 3: Test Milvus insertion ===")
milvus_service = MilvusService()
if result.get('success') and chunks:
    print("Testing insertion of first chunk...")
    chunk = chunks[0]
    milvus_service.insert_text(
        collection_name="documents",
        text=chunk.get('content', ''),
        document_id=str(doc.id),
        page=chunk.get('page', 0),
        metadata={"file_name": doc.file_name}
    )
    print("Insertion complete")

print("\n=== Step 4: Test search ===")
results = milvus_service.search_text("documents", "汽车构造", limit=3)
print(f"Search results: {len(results)}")
for i, r in enumerate(results):
    print(f"Result {i+1}: {r.get('text', '')[:100]}...")

session.close()
print("\n=== Test complete ===")
