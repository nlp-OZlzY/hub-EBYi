"""
多模态检索接口
- 文本搜多模态数据
- 按条件检索
"""
from fastapi import APIRouter, Query
from typing import Optional, List
import random
from core.response import ResponseModel

router = APIRouter()

# 模拟检索数据库
MOCK_RETRIEVAL_DB = [
    {"id": 1, "text": "这是一个关于深度学习的文档片段", "image_url": None, "doc_name": "深度学习入门.pdf", "page": 1},
    {"id": 2, "text": "Transformer 架构广泛应用于 NLP 任务", "image_url": None, "doc_name": "Transformer.pdf", "page": 5},
    {"id": 3, "text": "RAG 技术结合检索与生成提升问答质量", "image_url": None, "doc_name": "RAG 指南.pdf", "page": 12},
    {"id": 4, "text": "CLIP 模型实现图文跨模态检索", "image_url": None, "doc_name": "多模态模型.pdf", "page": 8},
    {"id": 5, "text": "BGE 模型用于中文文本向量编码", "image_url": None, "doc_name": "Embedding 模型.pdf", "page": 3},
]


@router.post("/retrieval/text")
async def retrieval_by_text(
    query: str = Query(...),
    limit: int = Query(5, ge=1, le=50)
):
    """文本搜多模态数据（模拟）"""
    # 模拟从 Milvus 检索结果
    shuffled = MOCK_RETRIEVAL_DB[:]
    random.shuffle(shuffled)
    results = shuffled[:min(limit, len(shuffled))]

    # 为每个结果添加随机相似度分数
    for item in results:
        item["score"] = round(random.uniform(0.7, 0.99), 4)

    return ResponseModel.success(data={
        "query": query,
        "total": len(results),
        "results": results
    }, msg="检索成功")


@router.post("/retrieval/condition")
async def retrieval_by_condition(
    query: str = Query(...),
    doc_type: Optional[str] = Query(None),
    file_id: Optional[int] = Query(None),
    limit: int = Query(5, ge=1, le=50)
):
    """按条件检索（模拟）"""
    filtered = MOCK_RETRIEVAL_DB[:]

    # 按文件ID过滤
    if file_id is not None:
        filtered = [item for item in filtered if item["id"] == file_id]

    # 按文档类型过滤（模拟）
    if doc_type:
        filtered = [item for item in filtered if doc_type in item.get("doc_name", "")]

    results = filtered[:min(limit, len(filtered))]

    for item in results:
        item["score"] = round(random.uniform(0.6, 0.95), 4)

    return ResponseModel.success(data={
        "query": query,
        "filters": {"doc_type": doc_type, "file_id": file_id},
        "total": len(results),
        "results": results
    }, msg="检索成功")