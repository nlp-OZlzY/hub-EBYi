"""Centralized configuration — loaded from environment variables with defaults."""

import os
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
DB_PATH = os.path.join(BASE_DIR, "db.db")

# ── Milvus (Zilliz Cloud) ──────────────────────────────────────────
MILVUS_URI: str = os.getenv(
    "MILVUS_URI",
    "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
)
MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "rag_data_new")

# ── Kafka ──────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "rag-data")

# ── DashScope (Qwen) ───────────────────────────────────────────────
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-plus")

# ── Model paths ────────────────────────────────────────────────────
BGE_MODEL_PATH: str = os.getenv(
    "BGE_MODEL_PATH", "/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5"
)
CLIP_MODEL_PATH: str = os.getenv(
    "CLIP_MODEL_PATH", "/root/autodl-tmp/models/jinaai/jina-clip-v2"
)

# ── Mineru ─────────────────────────────────────────────────────────
MINERU_BACKEND: str = os.getenv("MINERU_BACKEND", "vlm-http-client")
MINERU_ENDPOINT: str = os.getenv("MINERU_ENDPOINT", "http://127.0.0.1:8000")
MINERU_TIMEOUT: int = int(os.getenv("MINERU_TIMEOUT", "600"))

# ── Chunking ───────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "256"))

# ── RAG ────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

# ── HF mirror (for mainland China) ─────────────────────────────────
HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ── Supported file types ───────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

# ── Ensure directories exist ───────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
