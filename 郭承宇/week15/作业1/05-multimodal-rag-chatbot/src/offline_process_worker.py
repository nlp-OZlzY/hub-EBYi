"""Offline worker: consumes Kafka messages, parses PDFs via mineru, encodes chunks, inserts into Milvus."""

import glob
import json
import logging
import os
import traceback
from typing import Any

import numpy as np
from kafka import KafkaConsumer
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH,
    CHUNK_SIZE,
    CLIP_MODEL_PATH,
    HF_ENDPOINT,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    MILVUS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    MINERU_BACKEND,
    MINERU_ENDPOINT,
    MINERU_TIMEOUT,
    PROCESSED_DIR,
)
from orm_model import FileRecord, Session

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── HF mirror ──────────────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = HF_ENDPOINT


# ── Models ─────────────────────────────────────────────────────────
def load_models() -> tuple[SentenceTransformer, SentenceTransformer]:
    logger.info("Loading BGE model from %s ...", BGE_MODEL_PATH)
    bge_model = SentenceTransformer(BGE_MODEL_PATH)
    logger.info("BGE model loaded.")

    logger.info("Loading CLIP model from %s ...", CLIP_MODEL_PATH)
    clip_model = SentenceTransformer(CLIP_MODEL_PATH, trust_remote_code=True, truncate_dim=1024)
    logger.info("CLIP model loaded.")
    return bge_model, clip_model


# ── Text splitting ─────────────────────────────────────────────────
def split_text2chunks(lines: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split markdown lines into chunks ≤ chunk_size characters, preserving image refs."""
    chunks: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue
        if not chunks:
            chunks.append(line)
        elif len(chunks[-1]) <= chunk_size:
            chunks[-1] += "\n" + line
        else:
            chunks.append(line)
    return chunks


# ── Encoding ───────────────────────────────────────────────────────
def encode_text_and_image(
    text: str,
    markdown_path: str,
    bge_model: SentenceTransformer,
    clip_model: SentenceTransformer,
) -> tuple[list[float], list[float], list[float]]:
    """Encode a text chunk into BGE (512d), CLIP text (1024d), and CLIP image (1024d) vectors."""
    lines = text.split("\n")
    text_no_image = "\n".join([l for l in lines if not l.startswith("![")])
    image_lines = [l for l in lines if l.startswith("![")]

    # BGE text vector
    try:
        bge_vec = bge_model.encode(text_no_image, normalize_embeddings=True).tolist()
    except Exception:
        logger.warning("BGE encoding failed, using zeros. Error: %s", traceback.format_exc())
        bge_vec = np.zeros(512).tolist()

    # CLIP text vector
    try:
        clip_text_vec = clip_model.encode(text_no_image, normalize_embeddings=True).tolist()
    except Exception:
        logger.warning("CLIP text encoding failed, using zeros. Error: %s", traceback.format_exc())
        clip_text_vec = np.zeros(1024).tolist()

    # CLIP image vector
    if image_lines:
        img_rel = image_lines[0].split("](")[1].split(")")[0]
        img_real = os.path.dirname(markdown_path) + os.sep + os.path.basename(img_rel)
        try:
            logger.info("Encoding image: %s", img_real)
            clip_img_vec = clip_model.encode(img_real, normalize_embeddings=True).tolist()
        except Exception:
            logger.warning("CLIP image encoding failed, using zeros. Error: %s", traceback.format_exc())
            clip_img_vec = np.zeros(1024).tolist()
    else:
        clip_img_vec = np.zeros(1024).tolist()

    return bge_vec, clip_text_vec, clip_img_vec


# ── Document encoding ──────────────────────────────────────────────
def encode_document(
    path: str,
    file_id: int,
    file_name: str,
    file_path: str,
    bge_model: SentenceTransformer,
    clip_model: SentenceTransformer,
    milvus_client: Any,
) -> None:
    """Read markdown, split into chunks, encode each, insert into Milvus."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = split_text2chunks(lines)

    for chunk in chunks:
        try:
            bge_vec, clip_text_vec, clip_img_vec = encode_text_and_image(
                chunk, path, bge_model, clip_model
            )
            data = [{
                "text_vector": bge_vec,
                "clip_text_vector": clip_text_vec,
                "clip_image_vector": clip_img_vec,
                "text": chunk,
                "db_id": file_id,
                "file_name": file_name,
                "file_path": file_path,
            }]
            milvus_client.insert(collection_name=MILVUS_COLLECTION, data=data)
        except Exception:
            logger.error("Failed to insert chunk for file_id=%d: %s", file_id, traceback.format_exc())

    logger.info("Document encoded: file_id=%d, chunks=%d", file_id, len(chunks))


# ── Main loop ──────────────────────────────────────────────────────
def main() -> None:
    bge_model, clip_model = load_models()

    from pymilvus import MilvusClient

    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    logger.info("Milvus client connected.")

    # Ensure collection exists with correct schema
    if not milvus_client.has_collection(collection_name=MILVUS_COLLECTION):
        from pymilvus import CollectionSchema, FieldSchema, DataType

        schema = CollectionSchema(fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text_vector", DataType.FLOAT_VECTOR, dim=512),
            FieldSchema("clip_text_vector", DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema("clip_image_vector", DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("db_id", DataType.INT64),
            FieldSchema("file_name", DataType.VARCHAR, max_length=512),
            FieldSchema("file_path", DataType.VARCHAR, max_length=1024),
        ], auto_id=True)
        milvus_client.create_collection(
            collection_name=MILVUS_COLLECTION,
            schema=schema,
        )
        logger.info("Created Milvus collection '%s'.", MILVUS_COLLECTION)
    else:
        logger.info("Milvus collection '%s' already exists.", MILVUS_COLLECTION)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="rag-parser-group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_interval_ms=600000,
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    logger.info("Kafka consumer ready, listening on topic '%s'.", KAFKA_TOPIC)

    for msg in consumer:
        file_id: int | None = None
        try:
            payload: dict[str, Any] = msg.value
            logger.info("Received message: %s", payload)

            file_name: str = payload["file_name"]
            file_path: str = payload["file_path"]
            file_id = payload["id"]

            # Update state → 解析中
            with Session() as session:
                rec = session.query(FileRecord).filter(FileRecord.id == file_id).first()
                if rec:
                    rec.filestate = "解析中"
                    session.commit()

            if not os.path.exists(file_path):
                logger.warning("File not found on disk: %s", file_path)
                with Session() as session:
                    rec = session.query(FileRecord).filter(FileRecord.id == file_id).first()
                    if rec:
                        rec.filestate = "解析失败"
                        session.commit()
                continue

            # Run mineru via HTTP API (mineru-api server with VLM preloaded)
            import requests

            api_url = f"{MINERU_ENDPOINT}/parse"
            logger.info("Calling mineru API: %s for file %s", api_url, file_path)
            with open(file_path, "rb") as f:
                resp = requests.post(
                    api_url,
                    files={"file": (os.path.basename(file_path), f)},
                    data={"output_dir": PROCESSED_DIR},
                    timeout=MINERU_TIMEOUT,
                )
            if resp.status_code != 200:
                raise RuntimeError(f"mineru API returned {resp.status_code}: {resp.text}")

            # Find the generated markdown
            base_name = os.path.basename(file_path).split(".")[0]
            md_files = glob.glob(os.path.join(PROCESSED_DIR, base_name, "**", "*.md"), recursive=True)
            if not md_files:
                logger.error("No markdown output found for %s", file_name)
                with Session() as session:
                    rec = session.query(FileRecord).filter(FileRecord.id == file_id).first()
                    if rec:
                        rec.filestate = "解析失败"
                        session.commit()
                continue

            # Encode and insert
            encode_document(md_files[0], file_id, file_name, file_path, bge_model, clip_model, milvus_client)

            # Update state → 已完成
            with Session() as session:
                rec = session.query(FileRecord).filter(FileRecord.id == file_id).first()
                if rec:
                    rec.filestate = "已完成"
                    session.commit()

            logger.info("Successfully processed file_id=%d (%s)", file_id, file_name)
            consumer.commit()
            logger.info("Offset committed for file_id=%d", file_id)

        except Exception:
            logger.error("Error processing message: %s", traceback.format_exc())
            if file_id is not None:
                try:
                    with Session() as session:
                        rec = session.query(FileRecord).filter(FileRecord.id == file_id).first()
                        if rec:
                            rec.filestate = "解析失败"
                            session.commit()
                except Exception:
                    logger.error("Failed to update failure state for file_id=%d", file_id)
            consumer.commit()
            logger.warning("Offset committed despite error for file_id=%d (marked as 解析失败)", file_id)


if __name__ == "__main__":
    main()
