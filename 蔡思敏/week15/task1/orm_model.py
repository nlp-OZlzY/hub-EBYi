"""
多模态RAG聊天机器人 - 数据模型
支持知识库管理、文档上传、解析状态跟踪
"""
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text,
    Boolean, Float, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum

Base = declarative_base()

# ==================== 枚举定义 ====================
class FileState(enum.Enum):
    """文件处理状态"""
    UPLOADED = "uploaded"           # 已上传，待处理
    PARSING = "parsing"             # 解析中
    PARSED = "parsed"               # 已解析
    PARSE_FAILED = "parse_failed"     # 解析失败
    INDEXING = "indexing"           # 建索引中
    INDEXED = "indexed"            # 已索引
    INDEX_FAILED = "index_failed"  # 索引失败

class ChunkType(enum.Enum):
    """Chunk类型"""
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"  # 图文混合

# ==================== 知识库表 ====================
class KnowledgeBase(Base):
    """知识库表"""
    __tablename__ = 'knowledge_bases'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)  # 知识库名称
    description = Column(Text, nullable=True)                 # 描述
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 关联
    files = relationship("File", back_populates="knowledge_base")

    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, name='{self.name}')>"

# ==================== 文件表 ====================
class File(Base):
    """文件表 - 存储上传的文档"""
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)           # 原始文件名
    filepath = Column(String(1000), nullable=False)          # 存储路径
    file_size = Column(Integer, nullable=True)              # 文件大小(字节)
    file_hash = Column(String(64), nullable=True)          # 文件MD5哈希
    mime_type = Column(String(100), nullable=True)         # MIME类型

    # 状态
    state = Column(String(20), nullable=False, default=FileState.UPLOADED.value)
    state_message = Column(Text, nullable=True)            # 状态详情/错误信息

    # 解析结果
    parsed_path = Column(String(1000), nullable=True)      # 解析后文件路径(markdown)
    page_count = Column(Integer, nullable=True)            # 页数

    # 关联
    knowledge_base_id = Column(Integer, ForeignKey('knowledge_bases.id'), nullable=True)
    knowledge_base = relationship("KnowledgeBase", back_populates="files")
    chunks = relationship("Chunk", back_populates="file")

    # 时间戳
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<File(id={self.id}, filename='{self.filename}', state='{self.state}')>"

# ==================== Chunk表 ====================
class Chunk(Base):
    """Chunk表 - 文档切分后的内容块"""
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)                 # 文本内容
    content_type = Column(String(20), default=ChunkType.TEXT.value)

    # 向量维度(用于验证)
    bge_dim = Column(Integer, nullable=True)               # BGE向量维度
    clip_text_dim = Column(Integer, nullable=True)        # CLIP文本向量维度
    clip_image_dim = Column(Integer, nullable=True)       # CLIP图像向量维度

    # 位置信息
    page_num = Column(Integer, nullable=True)             # 所在页码
    chunk_index = Column(Integer, nullable=False)         # 在文档中的顺序

    # 关联
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    file = relationship("File", back_populates="chunks")

    # 关联图片
    images = relationship("ChunkImage", back_populates="chunk")

    # 时间戳
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Chunk(id={self.id}, file_id={self.file_id}, index={self.chunk_index})>"

# ==================== Chunk图片关联表 ====================
class ChunkImage(Base):
    """Chunk图片关联表"""
    __tablename__ = 'chunk_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False)
    image_path = Column(String(1000), nullable=False)     # 图片路径
    image_index = Column(Integer, default=0)              # 在chunk中的顺序

    chunk = relationship("Chunk", back_populates="images")

    def __repr__(self):
        return f"<ChunkImage(id={self.id}, chunk_id={self.chunk_id})>"

# ==================== 问答会话表 ====================
class ChatSession(Base):
    """问答会话表"""
    __tablename__ = 'chat_sessions'

    id = Column(String(36), primary_key=True)             # UUID
    knowledge_base_id = Column(Integer, ForeignKey('knowledge_bases.id'), nullable=False)

    # 统计信息
    message_count = Column(Integer, default=0)           # 消息数

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    messages = relationship("ChatMessage", back_populates="session")

    def __repr__(self):
        return f"<ChatSession(id='{self.id}', kb_id={self.knowledge_base_id})>"

class ChatMessage(Base):
    """聊天消息表"""
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey('chat_sessions.id'), nullable=False)

    role = Column(String(20), nullable=False)             # user/assistant
    content = Column(Text, nullable=False)                # 消息内容

    # 引用来源(assistant消息)
    source_chunks = Column(Text, nullable=True)           # JSON格式，关联的chunk ids

    created_at = Column(DateTime, default=datetime.now)

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}')>"

# ==================== 数据库初始化 ====================
def get_db_path():
    """获取数据库路径"""
    return os.path.join(os.getcwd(), 'multimodal_rag.db')

def init_db(db_path=None):
    """初始化数据库"""
    if db_path is None:
        db_path = get_db_path()

    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session, engine

# 导出
__all__ = [
    'Base', 'KnowledgeBase', 'File', 'Chunk', 'ChunkImage',
    'ChatSession', 'ChatMessage', 'FileState', 'ChunkType',
    'init_db', 'get_db_path'
]