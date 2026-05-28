"""
多模态RAG聊天机器人 - API接口规范
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import enum

# ==================== 枚举定义 ====================
class FileState(str, enum.Enum):
    """文件处理状态"""
    UPLOADED = "uploaded"           # 已上传，待处理
    PARSING = "parsing"             # 解析中
    PARSED = "parsed"               # 已解析
    PARSE_FAILED = "parse_failed"   # 解析失败
    INDEXING = "indexing"           # 建索引中
    INDEXED = "indexed"             # 已索引
    INDEX_FAILED = "index_failed"   # 索引失败

class ChunkType(str, enum.Enum):
    """Chunk类型"""
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"

# ==================== 知识库接口 ====================
class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=255, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")

class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class KnowledgeBaseList(BaseModel):
    """知识库列表响应"""
    items: List[KnowledgeBaseResponse]
    total: int

# ==================== 文件上传接口 ====================
class FileUploadRequest(BaseModel):
    """文件上传请求(用于内部传递)"""
    knowledge_base_id: Optional[int] = Field(None, description="知识库ID")
    file_name: str
    file_path: str
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    mime_type: Optional[str] = None

class FileResponse(BaseModel):
    """文件响应"""
    id: int
    filename: str
    filepath: str
    file_size: Optional[int]
    state: str
    state_message: Optional[str]
    knowledge_base_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class FileListResponse(BaseModel):
    """文件列表响应"""
    items: List[FileResponse]
    total: int

# ==================== 文档解析接口 ====================
class DocumentParseRequest(BaseModel):
    """文档解析请求"""
    file_id: int = Field(..., description="文件ID")
    parse_options: Optional[dict] = Field(None, description="解析选项")

class DocumentParseResponse(BaseModel):
    """文档解析响应"""
    file_id: int
    state: str
    parsed_path: Optional[str] = None
    page_count: Optional[int] = None
    message: Optional[str] = None

# ==================== 问答接口 ====================
class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")
    source_chunks: Optional[List[int]] = Field(None, description="引用的chunk IDs")

class ChatRequest(BaseModel):
    """聊天请求"""
    knowledge_base_id: int = Field(..., description="知识库ID")
    session_id: Optional[str] = Field(None, description="会话ID，为空则创建新会话")
    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索数量")

class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    message: ChatMessage
    retrieved_chunks: List[dict] = Field(default_factory=list, description="检索到的chunk")

class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""
    session_id: str
    messages: List[ChatMessage]
    total: int

# ==================== Chunk查询接口 ====================
class ChunkResponse(BaseModel):
    """Chunk响应"""
    id: int
    content: str
    content_type: str
    page_num: Optional[int]
    chunk_index: int
    file_id: int
    images: List[str] = Field(default_factory=list, description="关联的图片路径")

    class Config:
        from_attributes = True

class ChunkSearchRequest(BaseModel):
    """Chunk搜索请求"""
    knowledge_base_id: int
    query: str = Field(..., description="搜索query")
    content_type: Optional[ChunkType] = Field(None, description="过滤类型")
    limit: int = Field(10, ge=1, le=100)

class ChunkSearchResponse(BaseModel):
    """Chunk搜索响应"""
    items: List[ChunkResponse]
    total: int
    query: str

# ==================== 状态查询接口 ====================
class FileStatusResponse(BaseModel):
    """文件状态响应"""
    file_id: int
    state: str
    state_message: Optional[str]
    progress: Optional[float] = Field(None, description="进度 0-1")
    parsed_path: Optional[str]
    page_count: Optional[int]
    chunk_count: Optional[int]

# ==================== 错误响应 ====================
class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None

# ==================== API路由说明 ====================
"""
API路由设计：

POST   /api/v1/knowledge-bases           # 创建知识库
GET    /api/v1/knowledge-bases           # 获取知识库列表
GET    /api/v1/knowledge-bases/{id}      # 获取知识库详情

POST   /api/v1/files/upload              # 上传文件
GET    /api/v1/files                     # 获取文件列表
GET    /api/v1/files/{id}                # 获取文件详情
DELETE /api/v1/files/{id}                # 删除文件
GET    /api/v1/files/{id}/status         # 获取文件处理状态

POST   /api/v1/files/{id}/parse          # 触发文档解析
POST   /api/v1/files/{id}/reparse        # 重新解析文档

GET    /api/v1/chunks/search             # 搜索Chunks
GET    /api/v1/files/{id}/chunks         # 获取某个文件的chunks

POST   /api/v1/chat                      # 问答
GET    /api/v1/chat/sessions/{session_id}/history  # 获取聊天历史

GET    /api/v1/health                    # 健康检查
"""