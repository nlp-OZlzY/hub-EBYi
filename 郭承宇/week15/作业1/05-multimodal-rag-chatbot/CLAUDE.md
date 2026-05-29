# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal RAG (Retrieval-Augmented Generation) chatbot — accepts PDF uploads, parses them into markdown with images, encodes chunks as vectors, and answers questions by retrieving both text and images. Built as a course project.

## Project Structure

```
├── src/
│   ├── __init__.py                # Package marker
│   ├── config.py                  # All config from .env, with defaults
│   ├── orm_model.py               # SQLAlchemy FileRecord model + SQLite engine
│   ├── api_server.py              # FastAPI REST server (5 endpoints)
│   ├── offline_process_worker.py  # Kafka consumer: PDF parse → chunk → encode → Milvus
│   ├── web_demo.py                # Streamlit entry point (multi-page navigation)
│   ├── web_page_upload.py         # Streamlit page: file upload & management
│   └── web_page_chat.py           # Streamlit page: RAG chat with image rendering
├── models/                        # Local model files (gitignored)
│   ├── bge-small-zh-v1.5/
│   └── jina-clip-v2/
├── tests/
│   ├── conftest.py                # sys.path setup, heavy dependency mocking
│   ├── test_orm_model.py          # FileRecord CRUD tests
│   ├── test_text_split.py         # split_text2chunks tests
│   ├── test_encode.py             # encode_text_and_image tests
│   ├── test_upload_api.py         # POST /upload/document tests
│   ├── test_chat_api.py           # POST /chat tests
│   ├── test_file_management_api.py # GET/DELETE /files tests
│   └── test_integration.py        # End-to-end flow tests
├── requirements.txt
├── .env.example
└── .gitignore
```

## Architecture

Four components, three independent processes connected via Kafka:

```
src/web_page_upload.py ──(Kafka: "rag-data")──> src/offline_process_worker.py
       |                                                   |
  SQLite (src/orm_model.py)                          Milvus (Zilliz Cloud)

src/web_page_chat.py ──> Milvus ──> Qwen (DashScope)

src/api_server.py ──> FastAPI REST API (upload / chat / files / health)
```

### 1. Upload (`src/web_page_upload.py`) + (`src/api_server.py POST /upload/document`)
- Saves uploaded PDF/DOCX/TXT to `./uploads/`
- Inserts a row into SQLite `files` table (state: `已上传`)
- Publishes a JSON message to Kafka topic `rag-data`
- Falls back to state `待发送` if Kafka is unreachable

### 2. Offline Worker (`src/offline_process_worker.py`)
- Consumes Kafka messages (manual commit, group: `rag-parser-group`)
- Updates DB state: `已上传` → `解析中` → `已完成` / `解析失败`
- Calls mineru-api HTTP service (`POST /parse`) to convert PDF to markdown + images
- Splits markdown into chunks (`split_text2chunks`, default ≤256 chars)
- Encodes each chunk: BGE (512d text), CLIP (1024d text), CLIP (1024d image)
- Inserts all vectors + text into Milvus collection `rag_data_new`

### 3. Chat (`src/web_page_chat.py`) + (`src/api_server.py POST /chat`)
- Encodes user query with BGE → 512d vector
- Searches Milvus `text_vector` field (ANN, top-K)
- Rewrites relative image paths (`images/` → `./processed/{doc}/vlm/images/`)
- Calls Qwen (DashScope `qwen-plus`) with a RAG prompt containing retrieved context
- Renders markdown response with inline images

### 4. API Server (`src/api_server.py`)
FastAPI on port 8000, 5 endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload/document` | Multipart file upload → save + DB + Kafka |
| `POST` | `/chat` | RAG chat (question + top_k → answer + sources) |
| `GET` | `/files` | List all uploaded files |
| `DELETE` | `/files/{file_id}` | Delete file + DB record + Milvus vectors |
| `GET` | `/health` | Health check |

## Models & Services

| Component | Model/Service | Dimension | Purpose |
|-----------|--------------|-----------|---------|
| Text embedding | `BAAI/bge-small-zh-v1.5` | 512d | Chunk & query encoding |
| Multimodal embedding | `jinaai/jina-clip-v2` (truncated) | 1024d | Cross-modal text+image encoding |
| LLM | DashScope `qwen-plus` | — | RAG answer generation |
| PDF parsing | `mineru-api` | — | PDF → markdown + extracted images |
| Vector DB | Milvus (Zilliz Cloud Serverless) | — | ANN search on `text_vector` |
| Message queue | Kafka | — | Async document processing |
| Metadata DB | SQLite | — | File records (upload state tracking) |

## SQLite Schema (`files` table via `src/orm_model.py`, class: `FileRecord`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `filename` | VARCHAR(255) | Original filename |
| `filepath` | VARCHAR(1000) | Local save path |
| `filestate` | VARCHAR(20) | `已上传` → `解析中` → `已完成` / `解析失败` / `待发送` |

## Milvus Schema (collection: `rag_data_new`)

| Field | Type | Purpose |
|-------|------|---------|
| `text_vector` | FLOAT_VECTOR(512) | BGE text embedding (primary retrieval) |
| `clip_text_vector` | FLOAT_VECTOR(1024) | CLIP text embedding (cross-modal) |
| `clip_image_vector` | FLOAT_VECTOR(1024) | CLIP image embedding (cross-modal) |
| `text` | VARCHAR | Chunk content (markdown) |
| `db_id` | INT64 | FK to `files.id` |
| `file_name` | VARCHAR | Original filename |
| `file_path` | VARCHAR | Original file path |

## Configuration

All settings in `.env` (copy from `.env.example`). See `src/config.py` for all variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_URI` | Zilliz Cloud URL | Milvus connection |
| `MILVUS_TOKEN` | — | Zilliz auth token |
| `MILVUS_COLLECTION` | `rag_data_new` | Collection name |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker |
| `KAFKA_TOPIC` | `rag-data` | Topic name |
| `DASHSCOPE_API_KEY` | — | DashScope API key |
| `QWEN_MODEL` | `qwen-plus` | LLM model |
| `BGE_MODEL_PATH` | `./models/bge-small-zh-v1.5` | Local BGE path |
| `CLIP_MODEL_PATH` | `./models/jina-clip-v2` | Local CLIP path |
| `MINERU_ENDPOINT` | `http://127.0.0.1:8000` | mineru-api URL |
| `MINERU_TIMEOUT` | `600` | PDF parse timeout (seconds) |
| `MINERU_BACKEND` | `vlm-http-client` | mineru backend type |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror (mainland China) |
| `CHUNK_SIZE` | `256` | Max chars per chunk |
| `DEFAULT_TOP_K` | `5` | Default retrieval count |
| `SUPPORTED_EXTENSIONS` | `{.pdf, .docx, .txt}` | Accepted upload types (hardcoded in config.py) |

## Running the Project

Prerequisites: **Kafka** on `localhost:9092`, **mineru-api** on `http://127.0.0.1:8000`, **Milvus** (Zilliz Cloud or Docker).

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with real credentials

# 3. Start services (in order)
python src/offline_process_worker.py   # Kafka consumer (runs forever)
python src/api_server.py               # FastAPI on :8000
streamlit run src/web_demo.py          # Streamlit on :8501
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests mock all external services (Milvus, Kafka, LLM, models) — no running infrastructure needed. The `tests/conftest.py` sets up `sys.path` for `src/` imports and pre-mocks heavy dependencies (`sentence_transformers`, `kafka`, `pymilvus`).

## Key Implementation Details

- **Kafka**: Manual offset commit (`enable_auto_commit=False`). Commits after successful processing AND after errors (to avoid infinite retry loops on bad messages). Used because PDF parsing is GPU-intensive and slow (~1 min/file) — upload is fast (producer), parsing is slow (consumer), Kafka decouples the two.
- **mineru**: Called via HTTP API (`POST /parse`), not CLI subprocess. The mineru-api server preloads the VLM model, so the worker just sends files over HTTP. Worker sends the file, mineru writes markdown + images to `./processed/`.
- **Image paths**: Stored in Milvus as relative (`images/xxx.png`). Rewritten at query time to `./processed/{file_basename}/vlm/images/xxx.png`.
- **Lazy singletons** (api_server.py): BGE model, Milvus client, Kafka producer loaded on first use.
- **Streamlit caching** (web_page_chat.py): `@st.cache_resource` on BGE, Milvus, Qwen clients.
- **CONFIG**: `BASE_DIR` is project root (2 levels up from `src/config.py`). `load_dotenv()` explicitly reads from `{BASE_DIR}/.env`.
- **Milvus collection**: Created explicitly with schema on worker startup (`has_collection` → `create_collection`). Schema defines all 7 fields with correct types and dimensions.
- **ORM naming**: The SQLAlchemy model is `FileRecord` (not `File`) to avoid shadowing `fastapi.File`.
