# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multimodal RAG (Retrieval-Augmented Generation) chatbot for PDF documents. The system accepts PDF uploads, parses them using Mineru, embeds text/images using BGE/CLIP models, stores vectors in Milvus, and answers user questions via Qwen-VL.

## Architecture

```
web_page_upload.py ──→ Kafka (rag-data topic) ──→ offline_precess_worker.py
                                                        │
                                                        ▼
                                                    Mineru parsing
                                                        │
                                                        ▼
                                            BGE/CLIP embedding → Milvus
                                                        │
web_page_chat.py ←──────────────────────────────────────┘
     │
     └─→ Milvus (vector search) → Qwen-VL → answer
```

## Key Components

**Web Interface (Streamlit)**
- `web_demo.py` — Main entry point with navigation
- `web_page_upload.py` — File upload page; uploads PDF to `uploads/`, produces message to Kafka topic `rag-data`
- `web_page_chat.py` — RAG chat page; embeds query with BGE, searches Milvus, renders answer with markdown/images

**Offline Worker**
- `offline_precess_worker.py` — Kafka consumer; calls Mineru to parse PDFs, splits text into chunks, embeds with BGE+CLIP, stores in Milvus collection `rag_data_new`

**Data Model**
- `orm_model.py` — SQLAlchemy `File` table (id, filename, filepath, filestate) using SQLite `db.db`

## Commands

```bash
# Start the Streamlit app (file upload + chat UI)
streamlit run web_demo.py

# Start the offline worker (consumes Kafka, processes documents)
python offline_precess_worker.py

# Dependencies (key packages)
pip install streamlit kafka-python pymilvus sentence-transformers openai
```

## External Services

- **Kafka**: localhost:9092, topic `rag-data`
- **Milvus**: Zilliz cloud (configured in code), collection `rag_data_new`
- **Mineru**: localhost:30000 (vlm-http-client mode)
- **Models**: BGE-small-zh-v1.5 and jina-clip-v2 loaded from `/root/autodl-tmp/models/`

## Data Flow

1. User uploads PDF via `web_page_upload.py` → saved to `uploads/` → record inserted into SQLite → message sent to Kafka
2. `offline_precess_worker.py` consumes the message → calls `mineru` CLI to parse PDF → outputs markdown+images to `processed/{filename}/` → splits markdown into chunks → embeds each chunk with BGE (text) and CLIP (text+image) → inserts vectors into Milvus
3. User asks question via `web_page_chat.py` → query embedded with BGE → Milvus ANN search → top-5 results passed to Qwen-plus with RAG prompt → answer rendered with images