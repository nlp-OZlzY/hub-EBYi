
"""Verify all components are working correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []

def log(msg):
    results.append(msg)
    print(msg)

log("=== Starting Verification ===")

# 1. Test Embedding Service
log("\n[1/5] Testing Embedding Service...")
try:
    from services.embedding_service import EmbeddingService
    embedding_service = EmbeddingService()
    log("  EmbeddingService initialized successfully")
    
    test_text = "汽车的发动机是核心部件"
    vec = embedding_service.embed(test_text)
    if vec and len(vec) == 1152:
        log(f"  [OK] Embedding generated: {len(vec)} dimensions")
        log(f"  [OK] First 3 values: {vec[:3]}")
    else:
        log(f"  [FAIL] Failed to generate valid embedding")
except Exception as e:
    log(f"  [FAIL] EmbeddingService error: {e}")
    import traceback
    log(f"  {traceback.format_exc()}")

# 2. Test Milvus Connection
log("\n[2/5] Testing Milvus Connection...")
try:
    from services.milvus_service import MilvusService
    milvus_service = MilvusService()
    client = milvus_service.client
    collections = client.list_collections()
    log(f"  [OK] Connected to Milvus")
    log(f"  [OK] Current collections: {collections}")
except Exception as e:
    log(f"  [FAIL] Milvus connection error: {e}")
    import traceback
    log(f"  {traceback.format_exc()}")

# 3. Test MinerU/PDF parsing
log("\n[3/5] Testing PDF Parsing...")
try:
    from services.mineru_service import MinerUService
    mineru_service = MinerUService()
    test_pdf = os.path.join(os.path.dirname(__file__), "uploads", "09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf")
    if os.path.exists(test_pdf):
        log(f"  Test PDF found: {test_pdf}")
        result = mineru_service.parse_document(test_pdf)
        if result.get('success'):
            chunks = result.get('data', {}).get('chunks', [])
            log(f"  [OK] PDF parsed successfully")
            log(f"  [OK] {len(chunks)} chunks extracted")
            if chunks:
                log(f"  [OK] First chunk preview: {chunks[0].get('content', '')[:100]}...")
        else:
            log(f"  [FAIL] PDF parse failed: {result.get('error')}")
    else:
        log(f"  Test PDF not found: {test_pdf}")
except Exception as e:
    log(f"  [FAIL] PDF parsing error: {e}")
    import traceback
    log(f"  {traceback.format_exc()}")

# 4. Test Database
log("\n[4/5] Testing Database...")
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models.orm_model import Base, Document
    db_path = "sqlite:///./multimodal_rag.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()
    docs = session.query(Document).all()
    log(f"  [OK] Database connected")
    log(f"  [OK] {len(docs)} documents in DB")
    for doc in docs:
        log(f"    - ID {doc.id}: {doc.file_name} ({doc.status.value})")
    session.close()
except Exception as e:
    log(f"  [FAIL] Database error: {e}")
    import traceback
    log(f"  {traceback.format_exc()}")

# 5. Write summary
log("\n[5/5] Writing summary...")
summary_file = os.path.join(os.path.dirname(__file__), "verification_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
log(f"  [OK] Summary written to {summary_file}")

log("\n=== Verification Complete ===")
