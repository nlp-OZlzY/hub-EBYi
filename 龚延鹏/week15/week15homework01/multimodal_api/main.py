"""
FastAPI 应用入口
"""
from fastapi import FastAPI
from routers import data_manager, retrieval, chat

app = FastAPI(title="Multimodal RAG API", version="1.0.0")

# 注册路由
app.include_router(data_manager.router, prefix="/api/v1", tags=["数据管理"])
app.include_router(retrieval.router, prefix="/api/v1", tags=["多模态检索"])
app.include_router(chat.router, prefix="/api/v1", tags=["多模态问答"])


@app.get("/")
def root():
    return {"message": "Multimodal RAG API"}