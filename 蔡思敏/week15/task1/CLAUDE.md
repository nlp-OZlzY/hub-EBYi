# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

多模态RAG聊天机器人，支持图文检索和问答。用户上传PDF文档，系统解析后建立向量索引，用户提问时检索相关文本+图片后调用大模型生成答案。

## 架构流程

```
上传文件 → Kafka消息 → Worker消费
                           ↓
                    minerU解析PDF → markdown + 图片
                           ↓
                    chunk划分 + BGE/CLIP编码
                           ↓
                       Milvus存储
                           ↓
用户提问 → API检索Milvus → 调用Qwen生成答案 → 前端展示
```

## 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据模型 | `orm_model.py` | SQLite表结构：KnowledgeBase、File、Chunk、ChatSession |
| API服务 | `main_api.py` | FastAPI (8000端口)，文件管理+问答接口 |
| 离线Worker | `offline_process_worker.py` | Kafka消费→minerU解析→向量编码→Milvus |
| 上传页面 | `web_page_upload.py` | Streamlit，文件上传+状态展示 |
| 问答页面 | `web_page_chat.py` | Streamlit，RAG检索+图文展示 |
| Web入口 | `web_demo.py` | Streamlit多页面导航 |

## 数据模型关系

- KnowledgeBase (1) → File (N) → Chunk (N) → ChunkImage (N)
- ChatSession (1) → ChatMessage (N)

File.state 流转: uploaded → parsing → parsed → indexing → indexed

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务 (8000端口)
python main_api.py

# 启动Worker（Kafka消费者）
python offline_process_worker.py

# 启动Web界面
streamlit run web_demo.py

# 运行单元测试
pytest tests/ -v

# Worker独立测试
python offline_process_worker.py --test split   # 测试chunk划分
python offline_process_worker.py --test encode  # 测试编码
python offline_process_worker.py --test all     # 全部测试
```

## 关键配置

- Milvus: `COLLECTION_NAME = "rag_data_new"` (见 main_api.py)
- Kafka topic: `rag-data`
- minerU服务: `http://127.0.0.1:30000`
- 模型路径: `/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5`

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/knowledge-bases` | POST/GET | 创建/获取知识库 |
| `/api/v1/files/upload` | POST | 上传文件 |
| `/api/v1/files/{id}/status` | GET | 获取文件处理状态 |
| `/api/v1/files/{id}/parse` | POST | 触发文档解析 |
| `/api/v1/chunks/search` | POST | 搜索相关Chunks |
| `/api/v1/chat` | POST | 问答 |

## 注意事项

1. API服务依赖Milvus（云端）和Kafka（本地），启动前确保依赖可用
2. Worker需要加载BGE和CLIP模型（约几分钟），启动较慢
3. 前端页面需要API服务运行才能正常工作
4. API Key等敏感信息通过环境变量传入，不要硬编码
