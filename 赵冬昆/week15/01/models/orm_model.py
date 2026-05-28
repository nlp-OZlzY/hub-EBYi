"""SQLAlchemy ORM Models for Document Management"""

from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class DocumentStatus(str, enum.Enum):
    """Document processing status enumeration"""
    UPLOADING = "上传中"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "失败"


class Document(Base):
    """Document metadata model for SQLite database"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    qiniu_key = Column(String(512), nullable=True)
    qiniu_url = Column(String(1024), nullable=True)
    status = Column(
        SQLEnum(DocumentStatus),
        default=DocumentStatus.UPLOADING,
        nullable=False
    )
    file_size = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, file_name='{self.file_name}', status='{self.status}')>"


class ChatSession(Base):
    """Chat session model for conversation tracking"""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=True)
    session_id = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ChatSession(id={self.id}, session_id='{self.session_id}')>"


class ChatMessage(Base):
    """Chat message model for storing conversation history"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(String(4096), nullable=False)
    sources = Column(String(1024), nullable=True)  # JSON string of sources
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}', session_id='{self.session_id}')>"