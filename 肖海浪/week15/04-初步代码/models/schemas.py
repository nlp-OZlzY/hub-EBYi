"""Pydantic 数据模型 - 请求/响应结构定义"""
from pydantic import BaseModel, Field
from typing import List, Optional


class FileResponse(BaseModel):
    """上传文件响应"""
    file_id: int
    filename: str
    filestate: str


class UploadResponse(BaseModel):
    """上传接口响应"""
    code: int = 200
    data: FileResponse


class SourceInfo(BaseModel):
    """检索来源信息"""
    file_name: str
    relevance: float


class ChatData(BaseModel):
    """问答数据"""
    answer: str
    sources: List[SourceInfo]


class ChatResponse(BaseModel):
    """问答接口响应"""
    code: int = 200
    data: ChatData


class ChatRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, description="用户问题")


class FileInfo(BaseModel):
    """文件信息"""
    id: int
    filename: str
    filepath: str
    filestate: str


class FileListResponse(BaseModel):
    """文件列表响应"""
    code: int = 200
    data: List[FileInfo]


class DeleteResponse(BaseModel):
    """删除响应"""
    code: int = 200
    message: str = "删除成功"


class ErrorResponse(BaseModel):
    """错误响应"""
    code: int
    message: str
