"""FastAPI server providing upload, chat, and file-management endpoints."""

import json
import logging
import os
import traceback
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH,
    DASHSCOPE_API_KEY,
    DEFAULT_TOP_K,
    HF_ENDPOINT,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    MILVUS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    QWEN_MODEL,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from orm_model import FileRecord, Session

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── HF mirror ──────────────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="Multimodal RAG API", version="1.0.0")

# ── Lazy-loaded singletons ─────────────────────────────────────────
bge_model: SentenceTransformer | None = None
milvus_client: Any = None
kafka_producer: Any = None


def get_bge_model() -> SentenceTransformer:
    global bge_model
    if bge_model is None:
        logger.info("Loading BGE model from %s ...", BGE_MODEL_PATH)
        bge_model = SentenceTransformer(BGE_MODEL_PATH)
        logger.info("BGE model loaded.")
    return bge_model


def get_milvus_client() -> Any:
    global milvus_client
    if milvus_client is None:
        from pymilvus import MilvusClient

        logger.info("Connecting to Milvus at %s ...", MILVUS_URI)
        milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Milvus connected.")
    return milvus_client


def get_kafka_producer() -> Any:
    global kafka_producer
    if kafka_producer is None:
        from kafka import KafkaProducer

        logger.info("Connecting to Kafka at %s ...", KAFKA_BOOTSTRAP_SERVERS)
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        logger.info("Kafka producer ready.")
    return kafka_producer


# ── Request / Response schemas ─────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    knowledge_base_id: str = Field(default="default")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


RAG_PROMPT = """基于资料回答用户的问题：{question}

相关资料: {related_content}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，需要将图放在合适的相关内容的位置。
"""


# ── Endpoints ──────────────────────────────────────────────────────


@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF/DOCX/TXT document for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is empty")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    save_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:
        f.write(content)

    with Session() as session:
        record = FileRecord(filename=file.filename, filepath=save_path, filestate="已上传")
        session.add(record)
        session.flush()
        record_id: int = record.id
        session.commit()

    kafka_warning: str | None = None
    try:
        producer = get_kafka_producer()
        producer.send(
            KAFKA_TOPIC,
            value={"file_name": file.filename, "file_path": save_path, "id": record_id},
        )
        producer.flush()
        logger.info("Kafka message sent for file_id=%d", record_id)
    except Exception:
        logger.warning("Kafka unavailable for file_id=%d: %s", record_id, traceback.format_exc())
        with Session() as session:
            rec = session.query(FileRecord).filter(FileRecord.id == record_id).first()
            if rec:
                rec.filestate = "待发送"
                session.commit()
        kafka_warning = "Kafka 不可达，文件已保存但未发送解析消息，状态标记为'待发送'"

    return {
        "status": "success",
        "file_id": record_id,
        "message": "文件上传成功，等待解析" if kafka_warning is None else kafka_warning,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """RAG chat: encode question, search Milvus, call Qwen for answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        model = get_bge_model()
        query_embedding = model.encode(req.question, normalize_embeddings=True).tolist()
    except Exception:
        logger.error("BGE encoding failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to encode question")

    try:
        client = get_milvus_client()
        results = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_embedding],
            limit=req.top_k,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"],
        )
    except Exception:
        logger.error("Milvus search failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to search Milvus")

    related_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for hit in results[0]:
        entity = hit["entity"]
        text = entity.get("text", "")
        file_path = entity.get("file_path", "")
        file_dir = os.path.basename(file_path).split(".")[0] if file_path else ""
        if file_dir:
            text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")
        related_parts.append(text)
        sources.append({
            "file_name": entity.get("file_name", ""),
            "db_id": entity.get("db_id", -1),
            "text": text[:500],
        })

    related_content = "\n".join(related_parts)
    prompt = RAG_PROMPT.format(question=req.question, related_content=related_content)

    try:
        import openai

        qwen_client = openai.OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = completion.choices[0].message.content or ""
    except Exception:
        logger.error("Qwen API call failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to call LLM")

    return ChatResponse(answer=answer, sources=sources)


@app.get("/files")
def list_files():
    """Return all uploaded files."""
    with Session() as session:
        files = session.query(FileRecord).all()
        return [
            {"id": f.id, "filename": f.filename, "filepath": f.filepath, "filestate": f.filestate}
            for f in files
        ]


@app.delete("/files/{file_id}")
def delete_file(file_id: int):
    """Delete a file record, its local file, and its Milvus vectors."""
    with Session() as session:
        record = session.query(FileRecord).filter(FileRecord.id == file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="File not found")
        filepath = record.filepath
        session.delete(record)
        session.commit()

    if os.path.exists(filepath):
        os.remove(filepath)

    try:
        client = get_milvus_client()
        client.delete(collection_name=MILVUS_COLLECTION, filter=f"db_id == {file_id}")
        logger.info("Deleted Milvus vectors for db_id=%d", file_id)
    except Exception:
        logger.warning("Failed to delete Milvus vectors for db_id=%d: %s", file_id, traceback.format_exc())

    return {"status": "success", "message": "文件删除成功"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
