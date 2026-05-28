"""Chat API Routes."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from services.embedding import get_embedding_service
from services.vectorstore import get_vector_store
from services.llm import get_llm_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    kb_id: Optional[int] = None
    limit: int = 5


class Source(BaseModel):
    text: str
    file_name: str
    file_path: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """多模态问答"""
    # 1. 向量化查询
    emb_service = get_embedding_service()
    query_vector = emb_service.encode_text(req.question)

    # 2. 构建过滤条件
    filter_expr = f"db_id == {req.kb_id}" if req.kb_id else None

    # 3. 检索
    vs = get_vector_store()
    results = vs.search(query_vector, limit=req.limit, filter_expr=filter_expr)

    # 4. 组装上下文
    related_content = ""
    sources = []
    for r in results:
        text = r["entity"]["text"]
        # 替换图片路径
        file_dir = r["entity"]["file_name"].split(".")[0]
        text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")

        related_content += text + "\n"

        sources.append(Source(
            text=r["entity"]["text"],
            file_name=r["entity"]["file_name"],
            file_path=r["entity"]["file_path"]
        ))

    # 5. 调用LLM生成答案
    llm_service = get_llm_service()
    answer = llm_service.chat(req.question, related_content)

    return ChatResponse(answer=answer, sources=sources)