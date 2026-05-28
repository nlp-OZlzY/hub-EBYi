
"""Check database status"""
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

print("\n=== 数据库中文档列表 ===")
documents = session.query(Document).all()
print(f"共 {len(documents)} 个文档")
for doc in documents:
    print(f"ID: {doc.id}, 文件名: {doc.file_name}, 状态: {doc.status.value}, 大小: {doc.file_size} bytes, 路径: {doc.file_path}")

session.close()
