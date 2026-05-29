"""Search API Routes."""
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional

from services.embedding import get_embedding_service
from services.vectorstore import get_vector_store

router = APIRouter(prefix="/search", tags=["search"])


class SearchResult(BaseModel):
    text: str
    file_name: str
    file_path: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="搜索查询"),
    limit: int = Query(5, ge=1, le=20),
    kb_id: Optional[int] = Query(None, description="知识库ID过滤")
):
    """多模态检索"""
    # 向量化
    emb_service = get_embedding_service()
    query_vector = emb_service.encode_text(q)

    # 构建过滤条件
    filter_expr = f"db_id == {kb_id}" if kb_id else None

    # 检索
    vs = get_vector_store()
    results = vs.search(query_vector, limit=limit, filter_expr=filter_expr)

    return SearchResponse(
        results=[
            SearchResult(
                text=r["entity"]["text"],
                file_name=r["entity"]["file_name"],
                file_path=r["entity"]["file_path"],
                score=r["distance"]
            )
            for r in results
        ],
        total=len(results)
    )