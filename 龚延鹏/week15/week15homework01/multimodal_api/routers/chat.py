"""
多模态问答接口
- 基础问答接口
- 基于检索结果的问答（占位）
"""
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, List
from core.response import ResponseModel

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    retrieval_context: Optional[List[dict]] = None


# 模拟问答知识库
MOCK_QA_PAIRS = {
    "深度学习": "深度学习是机器学习的一个分支，使用多层神经网络自动学习数据的层次化表示。",
    "Transformer": "Transformer 是一种基于自注意力机制的模型架构，由 Vaswani 等人在 2017 年提出。",
    "RAG": "RAG（检索增强生成）结合了检索系统和语言模型，通过外部知识库提升生成质量。",
    "CLIP": "CLIP 是 OpenAI 开发的图文跨模态模型，能够理解图像和文本的关系。",
    "embedding": "Embedding 是将离散数据（如文字、图片）映射到连续向量空间的技术，便于计算相似度。",
}


@router.post("/chat/basic")
async def basic_chat(query: str = Query(...)):
    """基础问答接口（模拟）"""
    # 简单匹配模拟
    answer = "感谢您的提问，这是一个复杂的问题需要综合分析。"
    for key, value in MOCK_QA_PAIRS.items():
        if key in query:
            answer = value
            break

    return ResponseModel.success(data={
        "query": query,
        "answer": answer,
        "model": "qwen-plus (mock)"
    }, msg="问答成功")


@router.post("/chat/with_retrieval")
async def chat_with_retrieval(request: ChatRequest):
    """基于检索结果的问答（模拟）"""
    context = request.retrieval_context or []

    if context:
        # 基于检索上下文生成答案（模拟）
        context_summary = " | ".join([
            f"[{item.get('doc_name', 'unknown')}] {item.get('text', '')[:50]}..."
            for item in context[:3]
        ])
        answer = f"根据检索到的资料回答：{context_summary}"
    else:
        answer = "未找到相关上下文信息，请尝试其他问题。"

    sources = [item.get("doc_name", "unknown") for item in context if item.get("doc_name")]

    return ResponseModel.success(data={
        "query": request.query,
        "answer": answer,
        "sources": sources,
        "context_count": len(context),
        "model": "qwen-vl (mock)"
    }, msg="问答成功")