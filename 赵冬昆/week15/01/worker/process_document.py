"""Direct Document Processing Script (without Kafka)"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.orm_model import Base, Document, DocumentStatus
from services.mineru_service import MinerUService
from services.milvus_service import MilvusService


def process_document(document_id: int):
    """
    Process a document directly without Kafka.
    
    Args:
        document_id: ID of the document to process
        
    Returns:
        bool: True if processing succeeded
    """
    db_path = "sqlite:///./multimodal_rag.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    milvus_service = MilvusService()
    mineru_service = MinerUService()
    
    try:
        document = session.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"Document {document_id} not found")
            return False
            
        if document.status != DocumentStatus.UPLOADING:
            print(f"Document {document_id} status is {document.status}, skipping")
            return False
            
        print(f"Processing document: {document.file_name} (ID: {document_id})")
        print(f"Local path: {document.file_path}")

        document.status = DocumentStatus.PROCESSING
        session.commit()

        if os.path.exists(document.file_path):
            result = mineru_service.parse_document(document.file_path)
        else:
            print(f"Local file not found: {document.file_path}")
            return False
        print(f"MinerU result: {result.get('success', False)}")
        
        if result.get("success"):
            chunks = result.get("data", {}).get("chunks", [])
            images = result.get("data", {}).get("images", [])
            
            print(f"Parsed {len(chunks)} text chunks and {len(images)} images")
            
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")
                page = chunk.get("page", 0)
                
                milvus_service.insert_text(
                    collection_name="documents",
                    text=content,
                    document_id=str(document_id),
                    page=page,
                    metadata={
                        "file_name": document.file_name,
                        "chunk_id": chunk.get("id", str(uuid.uuid4()))
                    }
                )
                print(f"Inserted chunk {i+1}/{len(chunks)}")
            
            for i, image in enumerate(images):
                milvus_service.insert_image(
                    collection_name="document_images",
                    image_base64=image.get("base64", ""),
                    document_id=str(document_id),
                    page=image.get("page", 0),
                    metadata={
                        "file_name": document.file_name,
                        "image_id": image.get("id", str(uuid.uuid4()))
                    }
                )
                print(f"Inserted image {i+1}/{len(images)}")
            
            document.status = DocumentStatus.COMPLETED
            document.page_count = len(chunks)
            print(f"Document {document_id} processed successfully!")
            
        else:
            document.status = DocumentStatus.FAILED
            print(f"MinerU parsing failed: {result.get('error', 'Unknown error')}")
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        document = session.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = DocumentStatus.FAILED
            session.commit()
        print(f"Error processing document {document_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()


def list_documents():
    """List all documents in the database"""
    db_path = "sqlite:///./multimodal_rag.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()
    
    documents = session.query(Document).all()
    print(f"\n=== 数据库中文档列表 ===")
    print(f"共 {len(documents)} 个文档")
    for doc in documents:
        print(f"ID: {doc.id}, 文件名: {doc.file_name}, 状态: {doc.status.value}, 大小: {doc.file_size} bytes")
    
    session.close()
    return documents


if __name__ == "__main__":
    documents = list_documents()
    
    if not documents:
        print("\n没有找到文档，请先上传文档！")
        sys.exit(0)
    
    pending_docs = [doc for doc in documents if doc.status == DocumentStatus.UPLOADING]
    
    if not pending_docs:
        print("\n没有待处理的文档！")
        sys.exit(0)
    
    print(f"\n=== 自动处理 {len(pending_docs)} 个待处理文档 ===")
    
    for doc in pending_docs:
        print(f"\n处理文档: {doc.file_name} (ID: {doc.id})")
        success = process_document(doc.id)
        
        if success:
            print(f"✅ 文档 {doc.id} 处理完成！")
        else:
            print(f"❌ 文档 {doc.id} 处理失败！")
