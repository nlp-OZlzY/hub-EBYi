
"""Reset document status to UPLOADING"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.orm_model import Base, Document, DocumentStatus

db_path = "sqlite:///./multimodal_rag.db"
engine = create_engine(db_path, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Reset document 2 to UPLOADING status
doc = session.query(Document).filter(Document.id == 2).first()
if doc:
    doc.status = DocumentStatus.UPLOADING
    session.commit()
    print(f"Document 2 status reset to UPLOADING")
else:
    print("Document 2 not found")

session.close()
