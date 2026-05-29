"""Streamlit page: file management — upload, list, and delete documents."""

import json
import logging
import os
import uuid

import streamlit as st
from kafka import KafkaProducer
from pymilvus import MilvusClient

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    MILVUS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from orm_model import FileRecord, Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Clients ────────────────────────────────────────────────────────
milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


def get_kafka_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


# ── File operations ────────────────────────────────────────────────


def query_files() -> None:
    """Display all files with delete buttons."""
    with Session() as session:
        files = session.query(FileRecord).all()

    if not files:
        st.info("暂无文件记录。")
        return

    for file in files:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{file.filename}**")
        with col2:
            st.write(file.filestate)
        with col3:
            if st.button("删除", key=f"del_{file.id}"):
                delete_file(file.id)
                st.rerun()


def delete_file(file_id: int) -> None:
    """Delete a file record, its local file, and its Milvus vectors."""
    with Session() as session:
        record = session.query(FileRecord).filter(FileRecord.id == file_id).first()
        if not record:
            st.error("文件不存在")
            return
        filepath = record.filepath
        session.delete(record)
        session.commit()

    if os.path.exists(filepath):
        os.remove(filepath)

    try:
        milvus_client.delete(collection_name=MILVUS_COLLECTION, filter=f"db_id == {file_id}")
    except Exception:
        logger.warning("Failed to delete Milvus vectors for db_id=%d", file_id)

    st.success("文件删除成功")


# ── Page UI ────────────────────────────────────────────────────────

st.markdown("### 文件管理")
st.divider()
st.markdown("#### 已上传文件")
query_files()

st.divider()
st.markdown("#### 上传新文件")

extensions = [e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)]
uploaded_file = st.file_uploader("选择文件", type=extensions)

if uploaded_file is not None:
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()
    save_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    with Session() as session:
        record = FileRecord(filename=file_name, filepath=save_path, filestate="已上传")
        session.add(record)
        session.flush()
        record_id = record.id
        session.commit()

    try:
        producer = get_kafka_producer()
        producer.send(
            KAFKA_TOPIC,
            value={"file_name": file_name, "file_path": save_path, "id": record_id},
        )
        producer.flush()
        st.success(f"文件 '{file_name}' 上传成功，等待解析")
    except Exception:
        with Session() as session:
            rec = session.query(FileRecord).filter(FileRecord.id == record_id).first()
            if rec:
                rec.filestate = "待发送"
                session.commit()
        st.warning(f"文件 '{file_name}' 已保存，但 Kafka 不可达，状态标记为'待发送'")
