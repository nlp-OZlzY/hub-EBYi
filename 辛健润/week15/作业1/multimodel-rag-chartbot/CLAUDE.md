# CLAUDE.md

## 项目概览

多模态RAG Chatbot服务，实现PDF文档的多模态检索问答。

## 技术栈

- Python 3.10+ / FastAPI
- Milvus (Zilliz Cloud)
- BGE + CLIP 向量化
- Qwen-VL (阿里DashScope)

## 命令

- 启动服务: `uvicorn main:app --reload`
- 运行Worker: `python worker/processor.py`
- 测试: `pytest tests/`

