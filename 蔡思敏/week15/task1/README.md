# 多模态RAG聊天机器人

支持图文检索和问答的RAG系统。

## 项目结构

```
05-multimodal-rag-chatbot/
├── orm_model.py           # 数据模型（知识库、文件、Chunk、会话）
├── api_schemas.py         # API接口定义（Pydantic models）
├── main_api.py           # FastAPI服务（文件管理、问答接口）
├── offline_process_worker.py  # 离线处理Worker（Kafka消费、文档解析）
├── web_page_upload.py    # Streamlit上传页面
├── web_page_chat.py      # Streamlit问答页面
├── web_demo.py           # Web入口（多页面导航）
├── config.yaml           # 配置文件
├── requirements.txt      # 依赖
├── tests/                # 单元测试
└── README.md
```

## 核心模块

### 1. 数据模型 (orm_model.py)
- `KnowledgeBase`: 知识库
- `File`: 文件（上传、解析状态）
- `Chunk`: 文档切分块
- `ChunkImage`: Chunk关联图片
- `ChatSession`: 问答会话
- `ChatMessage`: 聊天消息

### 2. API接口 (main_api.py)

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/knowledge-bases` | POST/GET | 创建/获取知识库 |
| `/api/v1/files/upload` | POST | 上传文件 |
| `/api/v1/files/{id}/status` | GET | 获取文件状态 |
| `/api/v1/files/{id}/parse` | POST | 触发解析 |
| `/api/v1/chunks/search` | POST | 搜索Chunks |
| `/api/v1/chat` | POST | 问答 |
| `/api/v1/health` | GET | 健康检查 |

### 3. 离线处理 (offline_process_worker.py)
- 从Kafka消费文档解析任务
- 调用minerU解析PDF
- chunk划分 + BGE/CLIP向量编码
- 存储到Milvus

### 4. 前端页面
- `web_page_upload.py`: 文件上传、状态展示
- `web_page_chat.py`: RAG问答、图文展示

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动API服务
```bash
python main_api.py
```

### 3. 启动Worker
```bash
python offline_process_worker.py
```

### 4. 启动Web界面
```bash
streamlit run web_demo.py
```

## 测试

```bash
pytest tests/ -v
```

## 技术栈

- **API**: FastAPI + uvicorn
- **前端**: Streamlit
- **数据库**: SQLite (元数据) + Milvus (向量)
- **消息队列**: Kafka
- **模型**: BGE (文本向量) + CLIP (图文向量) + Qwen-VL (问答)