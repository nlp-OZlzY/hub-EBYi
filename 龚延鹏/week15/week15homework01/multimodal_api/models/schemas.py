"""
Pydantic 模型定义
"""
from pydantic import BaseModel
from typing import Optional, List


class RetrievalResult(BaseModel):
    id: int
    text: str
    score: float
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    context_count: int


class DataItem(BaseModel):
    id: int
    filename: Optional[str] = None
    content: Optional[str] = None
    type: str