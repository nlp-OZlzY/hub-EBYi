"""ORM Models for Multimodal RAG."""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum

Base = declarative_base()


class FileState(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "failed"
    FAILED = "failed"


class File(Base):
    """文件表"""
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    filestate = Column(Enum(FileState), default=FileState.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db_path():
    # 查找项目根目录
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, "db.db")


def init_db():
    db_path = get_db_path()
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


engine = init_db()
Session = sessionmaker(bind=engine)


def create_file_record(filename: str, filepath: str) -> File:
    """创建文件记录"""
    with Session() as session:
        f = File(filename=filename, filepath=filepath, filestate=FileState.PENDING)
        session.add(f)
        session.commit()
        session.refresh(f)
        return f


def get_file(id: int) -> File:
    """获取文件"""
    with Session() as session:
        return session.query(File).filter(File.id == id).first()


def list_files():
    """列出所有文件"""
    with Session() as session:
        return session.query(File).order_by(File.created_at.desc()).all()


def update_file_state(id: int, state: FileState):
    """更新文件状态"""
    with Session() as session:
        f = session.query(File).filter(File.id == id).first()
        if f:
            f.filestate = state
            session.commit()