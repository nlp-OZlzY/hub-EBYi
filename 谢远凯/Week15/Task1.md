# 作业1：多模态 RAG 对话系统 — Vibe Coding 实现

## 一、项目背景与需求

### 1.1 问题描述

传统问答系统仅处理纯文本，而真实世界的知识大量存在于图文混排的 PDF 文档中。用户可能提出如下问题：

- "这张图片里，左边那个设备的功能是什么？"
- "根据图表显示，产品A的销售额在哪个季度开始下降？"

这类问题需要模型同时理解自然语言和图像内容，进行跨模态推理。

### 1.2 系统目标

基于 `05-multimodal-rag-chatbot` 项目需求，实现一个**多模态 RAG 对话系统**，核心能力包括：

| 能力 | 说明 |
|------|------|
| 多模态信息理解 | 同时理解用户自然语言问题和知识库中的图像/文本内容 |
| 跨模态检索 | 从多个 PDF 文档中高效检索相关的图像、图表、文本段落 |
| 图文关联推理 | 将检索到的图像信息与文本信息融合，进行深层逻辑推理 |
| 答案生成 | 生成准确、简洁的答案，并指明信息来源（哪个 PDF、哪一页、哪个图表） |

### 1.3 评价方法

每个问题的综合评分（满分 1 分）：

- **页面匹配度**：0.25 分
- **文件名匹配度**：0.25 分
- **答案内容相似度**：0.5 分（Jaccard 相似系数 = 交集大小 / 并集大小）

### 1.4 技术栈

| 类别 | 技术选型 |
|------|----------|
| 编程语言 | Python 3.10+ |
| Web 框架 | FastAPI |
| 前端 | Streamlit（原型）|
| 文档解析 | MinerU（CLI/API）|
| 文本编码 | BGE-small-zh-v1.5（512 维）|
| 多模态编码 | Jina-CLIP-v2（1024 维）|
| 多模态问答 | Qwen-VL / Qwen-Plus |
| 向量数据库 | Milvus (Zilliz Cloud) |
| 元信息存储 | SQLite |
| 消息队列 | Kafka |

---

## 二、系统架构

### 2.1 整体架构图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Streamlit   │     │   FastAPI    │     │  Kafka Consumer      │
│  Web UI      │────▶│   Server     │────▶│  (offline_worker)    │
│              │     │              │     │                      │
│ - 文件上传   │     │ /upload/     │     │ 1. 消费 rag-data     │
│ - 图文对话   │     │   document   │     │ 2. 调用 MinerU 解析  │
│              │     │ /chat        │     │ 3. Chunk + Embedding │
│              │     │ /files       │     │ 4. 写入 Milvus       │
└──────────────┘     └──────────────┘     └──────────────────────┘
       │                    │                       │
       ▼                    ▼                       ▼
┌──────────────┐    ┌──────────────┐     ┌──────────────────────┐
│  Kafka       │    │   SQLite     │     │  Milvus              │
│  Producer    │    │   (元信息)   │     │  (向量存储)          │
│  rag-data    │    │              │     │                      │
└──────────────┘    └──────────────┘     └──────────────────────┘
```

### 2.2 模块划分

```
multimodal_rag/
├── api/                    # FastAPI 接口层
│   ├── __init__.py
│   ├── main.py             # FastAPI app 入口
│   ├── routes/
│   │   ├── upload.py       # POST /upload/document
│   │   ├── chat.py         # POST /chat
│   │   └── files.py        # GET /files, DELETE /files/{id}
│   └── schemas.py          # Pydantic 请求/响应模型
├── core/                   # 核心业务逻辑
│   ├── __init__.py
│   ├── document_parser.py  # MinerU 文档解析
│   ├── embedding.py        # BGE + CLIP 编码
│   ├── chunker.py          # Markdown 文本分块
│   ├── retriever.py        # Milvus 向量检索
│   └── generator.py        # Qwen-VL 问答生成
├── models/                 # 数据模型
│   ├── __init__.py
│   └── orm.py              # SQLAlchemy ORM
├── worker/                 # 离线处理 Worker
│   ├── __init__.py
│   └── consumer.py         # Kafka 消费者
├── web/                    # Streamlit 前端
│   ├── app.py              # 主入口
│   ├── page_upload.py      # 文件管理页
│   └── page_chat.py        # 图文对话页
├── config.py               # 配置管理
├── tests/                  # 测试
│   ├── test_chunker.py
│   ├── test_embedding.py
│   ├── test_api.py
│   └── test_retriever.py
└── requirements.txt
```

### 2.3 数据流

**文档上传流程：**

```
用户上传 PDF → API /upload/document
  → 保存文件到本地 (uploads/<uuid>.pdf)
  → 写入 SQLite (filename, filepath, filestate="已上传")
  → 发送 Kafka 消息 {file_name, file_path, id}
  → 返回 file_id
```

**离线解析流程：**

```
Kafka Consumer 收到消息
  → 调用 MinerU CLI: mineru -p <path> -o ./processed
  → 获取解析后的 Markdown + 图片
  → split_text2chunks() 切分为 ~256 字符块
  → encode_text_and_image() 编码 (BGE文本 + CLIP文本 + CLIP图像)
  → 插入 Milvus collection "rag_data_new"
  → 更新 SQLite filestate="已处理"
```

**问答流程：**

```
用户提问 → API /chat
  → BGE 编码用户提问
  → Milvus 向量检索 top-5 (text_vector 字段)
  → 图片路径重写 (相对路径 → 绝对路径)
  → 构造 RAG Prompt (问题 + 检索内容)
  → Qwen-Plus 生成回答
  → 渲染 Markdown (图文混排)
```

---

## 三、接口定义

### 3.1 POST /upload/document

上传文档到指定知识库。

**请求：** `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | PDF/DOCX/TXT 文件 |

**响应：**

```json
{
  "id": 1,
  "filename": "example.pdf",
  "filepath": "uploads/uuid.pdf",
  "filestate": "已上传",
  "message": "文件上传成功，正在后台解析"
}
```

**处理步骤：**
1. 保存文件到 `uploads/<uuid><ext>`
2. 插入 SQLite 记录（filestate="已上传"）
3. 发送 Kafka 消息到 `rag-data` topic
4. 返回文件信息

### 3.2 POST /chat

多模态 RAG 问答。

**请求：**

```json
{
  "question": "VLLM 的 Memory layout 是什么？",
  "top_k": 5
}
```

**响应：**

```json
{
  "answer": "VLLM 的 Memory layout 采用 PagedAttention...",
  "sources": [
    {
      "file_name": "2309 vllm.pdf",
      "db_id": 1,
      "text": "相关文本片段..."
    }
  ]
}
```

**处理步骤：**
1. BGE 编码用户提问
2. Milvus 检索 top-k 结果
3. 图片路径重写
4. 构造 RAG Prompt 调用 Qwen-Plus
5. 返回答案和来源

### 3.3 GET /files

获取所有文件列表。

**响应：**

```json
{
  "files": [
    {
      "id": 1,
      "filename": "example.pdf",
      "filepath": "uploads/uuid.pdf",
      "filestate": "已处理"
    }
  ]
}
```

### 3.4 DELETE /files/{file_id}

删除文件及其关联的向量数据。

**响应：**

```json
{
  "message": "文件删除成功",
  "deleted_vectors": 15
}
```

---

## 四、测试逻辑

### 4.1 文本分块测试 (test_chunker.py)

```python
import pytest
from core.chunker import split_text2chunks

def test_empty_input():
    """空输入返回空列表"""
    assert split_text2chunks([]) == []

def test_short_lines_merged():
    """短行应合并为一个 chunk"""
    lines = ["第一行", "第二行", "第三行"]
    chunks = split_text2chunks(lines, chunk_size=256)
    assert len(chunks) == 1
    assert "第一行" in chunks[0]
    assert "第二行" in chunks[0]

def test_long_line_split():
    """超过 chunk_size 的行应单独成块"""
    lines = ["a" * 300]
    chunks = split_text2chunks(lines, chunk_size=256)
    assert len(chunks) == 1  # 单行不拆分，但下一个短行会另起 chunk
    assert len(chunks[0]) == 300

def test_skip_references():
    """跳过 # References 和编号引用行"""
    lines = ["# References", "[1] Author et al., 2024", "正常内容"]
    chunks = split_text2chunks(lines, chunk_size=256)
    assert len(chunks) == 1
    assert "正常内容" in chunks[0]
    assert "References" not in chunks[0]

def test_skip_empty_lines():
    """跳过空行"""
    lines = ["内容A", "", "", "内容B"]
    chunks = split_text2chunks(lines, chunk_size=256)
    assert len(chunks) == 1
```

### 4.2 编码测试 (test_embedding.py)

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

def test_text_bge_embedding_dimension():
    """BGE 文本编码维度应为 512"""
    # Mock 模型，验证输出维度
    with patch("core.embedding.bge_model") as mock_model:
        mock_model.encode.return_value = np.random.rand(512)
        from core.embedding import encode_text
        vec = encode_text("测试文本")
        assert vec.shape == (512,)

def test_clip_text_embedding_dimension():
    """CLIP 文本编码维度应为 1024"""
    with patch("core.embedding.clip_model") as mock_model:
        mock_model.encode.return_value = np.random.rand(1024)
        from core.embedding import encode_clip_text
        vec = encode_clip_text("测试文本")
        assert vec.shape == (1024,)

def test_encode_text_and_image_fallback():
    """编码失败时返回零向量"""
    with patch("core.embedding.bge_model") as mock_bge, \
         patch("core.embedding.clip_model") as mock_clip:
        mock_bge.encode.side_effect = Exception("编码失败")
        mock_clip.encode.side_effect = Exception("编码失败")
        from core.embedding import encode_text_and_image
        bge_vec, clip_t, clip_i = encode_text_and_image("文本", "/path/to/md")
        assert np.all(bge_vec == 0)
        assert np.all(clip_t == 0)
        assert np.all(clip_i == 0)
```

### 4.3 API 接口测试 (test_api.py)

```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_upload_document():
    """测试文件上传接口"""
    with open("test_sample.pdf", "rb") as f:
        response = client.post(
            "/upload/document",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["filestate"] == "已上传"
    assert "id" in data

def test_get_files():
    """测试获取文件列表"""
    response = client.get("/files")
    assert response.status_code == 200
    assert "files" in response.json()

def test_chat_endpoint():
    """测试问答接口"""
    response = client.post(
        "/chat",
        json={"question": "什么是 VLLM？", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data

def test_delete_file():
    """测试删除文件"""
    # 先上传
    with open("test_sample.pdf", "rb") as f:
        upload_resp = client.post(
            "/upload/document",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    file_id = upload_resp.json()["id"]
    # 再删除
    response = client.delete(f"/files/{file_id}")
    assert response.status_code == 200
```

### 4.4 检索测试 (test_retriever.py)

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

def test_retriever_returns_top_k():
    """检索应返回 top_k 个结果"""
    with patch("core.retriever.client") as mock_client:
        mock_client.search.return_value = [
            {"entity": {"text": "chunk1", "db_id": 1, "file_name": "a.pdf", "file_path": "a.pdf"}, "distance": 0.9},
            {"entity": {"text": "chunk2", "db_id": 2, "file_name": "b.pdf", "file_path": "b.pdf"}, "distance": 0.8},
        ]
        from core.retriever import search
        results = search(np.random.rand(512).tolist(), top_k=2)
        assert len(results) <= 2

def test_retriever_image_path_rewrite():
    """图片路径应从相对路径转为绝对路径"""
    from core.retriever import rewrite_image_paths
    text = "![img](images/fig1.jpg)"
    result = rewrite_image_paths(text, "/data/processed/test_pdf/vlm")
    assert "images/" not in result
    assert "/data/processed/test_pdf/vlm/images/fig1.jpg" in result
```

---

## 五、核心代码实现

### 5.1 配置管理 — `config.py`

```python
import os

# --- 模型路径 ---
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "jinaai/jina-clip-v2")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")

# --- Milvus ---
MILVUS_URI = os.getenv("MILVUS_URI", "")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = "rag_data_new"

# --- Kafka ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = "rag-data"

# --- SQLite ---
DB_PATH = os.path.join(os.getcwd(), "db.db")

# --- 文件存储 ---
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
PROCESSED_DIR = os.path.join(os.getcwd(), "processed")

# --- 向量维度 ---
BGE_DIM = 512
CLIP_DIM = 1024

# --- MinerU ---
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://127.0.0.1:30000")
```

### 5.2 ORM 模型 — `models/orm.py`

```python
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    filestate = Column(String(20), nullable=False, default="已上传")


db_path = os.path.join(os.getcwd(), "db.db")
engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
```

### 5.3 API 请求/响应模型 — `api/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    file_name: str
    db_id: int
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class FileItem(BaseModel):
    id: int
    filename: str
    filepath: str
    filestate: str


class FileListResponse(BaseModel):
    files: List[FileItem]


class UploadResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    filestate: str
    message: str


class DeleteResponse(BaseModel):
    message: str
    deleted_vectors: int
```

### 5.4 FastAPI 主入口 — `api/main.py`

```python
from fastapi import FastAPI
from api.routes import upload, chat, files

app = FastAPI(title="Multimodal RAG Chatbot", version="1.0.0")

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(files.router, prefix="/files", tags=["files"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 5.5 上传接口 — `api/routes/upload.py`

```python
import os
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from kafka import KafkaProducer

from api.schemas import UploadResponse
from models.orm import File, Session
from config import UPLOAD_DIR, KAFKA_BOOTSTRAP, KAFKA_TOPIC

router = APIRouter()

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


@router.post("/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # 1. 保存文件
    file_ext = os.path.splitext(file.filename)[1]
    save_name = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, save_name + file_ext)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 2. 写入数据库
    with Session() as session:
        record = File(filename=file.filename, filepath=save_path, filestate="已上传")
        session.add(record)
        session.flush()
        record_id = record.id
        session.commit()

    # 3. 发送 Kafka 消息
    producer.send(
        KAFKA_TOPIC,
        value={"file_name": file.filename, "file_path": save_path, "id": record_id},
    )
    producer.flush()

    return UploadResponse(
        id=record_id,
        filename=file.filename,
        filepath=save_path,
        filestate="已上传",
        message="文件上传成功，正在后台解析",
    )
```

### 5.6 问答接口 — `api/routes/chat.py`

```python
from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse, SourceItem
from core.retriever import search, rewrite_image_paths
from core.generator import generate_answer

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. 检索
    results = search_by_text(req.question, top_k=req.top_k)

    # 2. 构造来源
    sources = []
    related_content = ""
    for r in results:
        text = rewrite_image_paths(r["entity"]["text"], r["entity"]["file_path"])
        sources.append(
            SourceItem(
                file_name=r["entity"]["file_name"],
                db_id=r["entity"]["db_id"],
                text=text,
            )
        )
        related_content += text + "\n"

    # 3. 生成答案
    answer = generate_answer(req.question, related_content)

    return ChatResponse(answer=answer, sources=sources)
```

### 5.7 文件管理接口 — `api/routes/files.py`

```python
from fastapi import APIRouter, HTTPException
from api.schemas import FileListResponse, FileItem, DeleteResponse
from models.orm import File, Session
from core.retriever import delete_vectors_by_file_id

router = APIRouter()


@router.get("/", response_model=FileListResponse)
async def list_files():
    with Session() as session:
        files = session.query(File).all()
    return FileListResponse(
        files=[
            FileItem(id=f.id, filename=f.filename, filepath=f.filepath, filestate=f.filestate)
            for f in files
        ]
    )


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: int):
    with Session() as session:
        file = session.query(File).filter(File.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")
        session.delete(file)
        session.commit()

    deleted_count = delete_vectors_by_file_id(file_id)
    return DeleteResponse(message="文件删除成功", deleted_vectors=deleted_count)
```

### 5.8 文本分块 — `core/chunker.py`

```python
from typing import List, Optional


def split_text2chunks(lines: List[str], chunk_size: int = 256) -> List[str]:
    """将文本行分割成多个块，每个块不超过 chunk_size 个字符"""
    chunks: List[str] = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        if line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if len(chunks) == 0:
            chunks.append(line)
        else:
            if len(chunks[-1]) <= chunk_size:
                chunks[-1] += "\n" + line
            else:
                chunks.append(line)

    return chunks


def split_markdown_by_headers(
    markdown_text: str, path: str, max_length: Optional[int] = 1024
) -> List[dict]:
    """按 Markdown 标题分割文本"""
    import re

    header_pattern = r"(^#+\s+.+$)"
    lines = markdown_text.split("\n")
    chunks = []
    current_chunk = []
    current_header = "Document"

    for line in lines:
        if re.match(header_pattern, line.strip()):
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "header": current_header, "path": path})
            current_chunk = [line]
            current_header = line.strip()
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "header": current_header, "path": path})

    # 按最大长度进一步分割
    if max_length:
        final_chunks = []
        for chunk in chunks:
            if len(chunk["text"]) <= max_length:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk["text"]), max_length):
                    final_chunks.append(
                        {
                            "text": chunk["text"][i : i + max_length],
                            "header": f"{chunk['header']} (Part {i // max_length + 1})",
                            "path": path,
                        }
                    )
        return final_chunks

    return chunks
```

### 5.9 向量编码 — `core/embedding.py`

```python
import os
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import BGE_MODEL_PATH, CLIP_MODEL_PATH, BGE_DIM, CLIP_DIM

# 延迟加载
_bge_model = None
_clip_model = None


def get_bge_model():
    global _bge_model
    if _bge_model is None:
        _bge_model = SentenceTransformer(BGE_MODEL_PATH)
    return _bge_model


def get_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(
            CLIP_MODEL_PATH, trust_remote_code=True, truncate_dim=CLIP_DIM
        )
    return _clip_model


def encode_text(text: str) -> np.ndarray:
    """BGE 文本编码"""
    model = get_bge_model()
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec
    except Exception:
        traceback.print_exc()
        return np.zeros(BGE_DIM)


def encode_clip_text(text: str) -> np.ndarray:
    """CLIP 文本编码"""
    model = get_clip_model()
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec
    except Exception:
        traceback.print_exc()
        return np.zeros(CLIP_DIM)


def encode_clip_image(image_path: str) -> np.ndarray:
    """CLIP 图像编码"""
    model = get_clip_model()
    try:
        vec = model.encode(image_path, normalize_embeddings=True)
        return vec
    except Exception:
        traceback.print_exc()
        return np.zeros(CLIP_DIM)


def encode_text_and_image(text: str, markdown_path: str):
    """同时编码文本和图像，返回 (bge_vec, clip_text_vec, clip_image_vec)"""
    text_with_no_image = "\n".join(
        [line for line in text.split("\n") if not line.startswith("![")]
    )
    text_with_image = [line for line in text.split("\n") if line.startswith("![")]

    bge_vec = encode_text(text_with_no_image)
    clip_text_vec = encode_clip_text(text_with_no_image)

    if len(text_with_image) > 0:
        image_path = text_with_image[0].split("](")[1].split(")")[0]
        image_real_path = os.path.dirname(markdown_path) + image_path.split("/")[-1]
        clip_image_vec = encode_clip_image(image_real_path)
    else:
        clip_image_vec = np.zeros(CLIP_DIM)

    return list(bge_vec), list(clip_text_vec), list(clip_image_vec)
```

### 5.10 向量检索 — `core/retriever.py`

```python
import os
from pymilvus import MilvusClient

from config import MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION, BGE_DIM
from core.embedding import encode_text

_client = None


def get_client():
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    return _client


def search_by_text(query: str, top_k: int = 5):
    """文本向量检索"""
    client = get_client()
    query_vec = encode_text(query)
    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[list(query_vec)],
        limit=top_k,
        anns_field="text_vector",
        output_fields=["text", "db_id", "file_name", "file_path"],
    )
    return results[0]


def rewrite_image_paths(text: str, file_path: str) -> str:
    """将 Markdown 中的相对图片路径转为绝对路径"""
    file_dir = os.path.basename(file_path).split(".")[0]
    return text.replace("images/", f"./processed/{file_dir}/vlm/images/")


def delete_vectors_by_file_id(file_id: int) -> int:
    """删除指定文件的所有向量"""
    client = get_client()
    result = client.delete(
        collection_name=MILVUS_COLLECTION, filter=f"db_id == {file_id}"
    )
    return result.get("delete_count", 0) if isinstance(result, dict) else 0
```

### 5.11 答案生成 — `core/generator.py`

```python
import openai

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_CHAT_MODEL

RAG_PROMPT = """基于资料回答的提问提问问题：{0}

相关资料: {1}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，需要将图放在合适的相关内容的位置。
"""

_client = None


def get_qwen_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    return _client


def generate_answer(question: str, context: str) -> str:
    """基于检索结果生成答案"""
    client = get_qwen_client()
    response = client.chat.completions.create(
        model=QWEN_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": RAG_PROMPT.format(question, context)},
        ],
    )
    return response.choices[0].message.content
```

### 5.12 文档解析 — `core/document_parser.py`

```python
import os
import glob
import subprocess

from config import PROCESSED_DIR, MINERU_API_URL


def parse_with_mineru(file_path: str) -> str | None:
    """使用 MinerU CLI 解析 PDF 文档，返回 Markdown 文件路径"""
    cmd = f"mineru -p {file_path} -o {PROCESSED_DIR} -b vlm-http-client -u {MINERU_API_URL}"
    try:
        subprocess.check_output(cmd, shell=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"MinerU 解析超时: {file_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"MinerU 解析失败: {e}")
        return None

    # 查找输出的 Markdown 文件
    basename = os.path.basename(file_path).split(".")[0]
    md_paths = glob.glob(
        os.path.join(PROCESSED_DIR, basename) + "/**/**.md"
    )
    return md_paths[0] if md_paths else None
```

### 5.13 Kafka 消费者 — `worker/consumer.py`

```python
import os
import json
import traceback

from kafka import KafkaConsumer

from config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, MILVUS_COLLECTION
from core.document_parser import parse_with_mineru
from core.chunker import split_text2chunks
from core.embedding import encode_text_and_image
from core.retriever import get_client

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


def encode_document(path: str, file_id: int, file_name: str, file_path: str):
    """读取 Markdown 文件，分块编码，写入 Milvus"""
    client = get_client()
    lines = open(path, "r", encoding="utf-8").readlines()
    chunks = split_text2chunks(lines)

    for chunk in chunks:
        try:
            bge_vec, clip_text_vec, clip_image_vec = encode_text_and_image(chunk, path)
            data = [
                {
                    "text_vector": bge_vec,
                    "clip_text_vector": clip_text_vec,
                    "clip_image_vector": clip_image_vec,
                    "text": chunk,
                    "db_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                }
            ]
            client.insert(collection_name=MILVUS_COLLECTION, data=data)
        except Exception:
            traceback.print_exc()


def main():
    """消费 Kafka 消息，解析文档并编码"""
    for msg in consumer:
        try:
            file_name = msg.value["file_name"]
            file_path = msg.value["file_path"]
            file_id = msg.value["id"]

            if not os.path.exists(file_path):
                continue

            md_path = parse_with_mineru(file_path)
            if md_path is None:
                print(f"解析失败: {file_name}")
                continue

            encode_document(md_path, file_id, file_name, file_path)
            print(f"处理完成: {file_name}")

        except Exception as e:
            print(f"处理出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

## 六、运行说明

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 FastAPI 服务
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动 Kafka 消费者（离线解析 Worker）
python -m worker.consumer

# 4. 启动 Streamlit 前端（可选）
streamlit run web/app.py

# 5. 运行测试
pytest tests/ -v
```

## 七、Vibe Coding 过程记录

本项目的开发过程遵循 **Vibe Coding** 模式：

1. **需求分析** → 阅读 `05-multimodal-rag-chatbot` 的 README 和源码，提取核心接口和流程
2. **架构设计** → 确定模块划分、数据流、技术选型，使用 Claude Code 辅助生成架构图和目录结构
3. **测试先行** → 先写测试逻辑（chunker/embedding/api/retriever），明确预期行为
4. **逐步实现** → Claude Code 逐模块生成代码，从 config → orm → schemas → routes → core → worker
5. **接口对齐** → 确保新实现兼容原始项目的接口（/upload/document, /chat, /files）

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 使用 FastAPI 替代纯 Streamlit | 原项目 README 指定 FastAPI，且更适合 API 化部署 |
| 保留 Kafka 解耦上传与解析 | MinerU 解析耗时（~1 min/file），必须异步处理 |
| 延迟加载模型 | 避免导入时加载大模型，加快启动速度 |
| 三向量存储（BGE+CLIP文本+CLIP图像）| 支持跨模态检索，与原项目保持一致 |
