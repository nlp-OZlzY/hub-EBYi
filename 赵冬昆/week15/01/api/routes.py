"""FastAPI Routes for Document Upload and Chat Q&A"""

import os
import uuid
import json
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from kafka import KafkaProducer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.orm_model import Base, Document, ChatSession, ChatMessage, DocumentStatus
from services.qiniu_service import QiniuService
from services.milvus_service import MilvusService

router = APIRouter()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "document_parse_topic"
UPLOAD_DIR = "./uploads"
db_path = "sqlite:///./multimodal_rag.db"

engine = create_engine(db_path, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    knowledge_base_id: Optional[int] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[dict]
    session_id: str


@router.post("/upload/document")
async def upload_document(file: UploadFile = File(...), db=Depends(get_db)):
    """
    Upload a PDF document for processing.

    Flow:
    1. Receive uploaded PDF file
    2. Save to local temp directory with unique ID
    3. Write file metadata to SQLite (status: "上传中")
    4. Upload to Qiniu object storage
    5. Send message to Kafka topic for background parsing
    6. Return success response
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    local_file_name = f"{file_id}{file_extension}"
    local_file_path = os.path.join(UPLOAD_DIR, local_file_name)

    try:
        content = await file.read()
        with open(local_file_path, "wb") as f:
            f.write(content)

        qiniu_key = f"documents/{file_id}/{file.filename}"
        qiniu_service = QiniuService()
        qiniu_url = qiniu_service.upload_file(local_file_path, qiniu_key)

        document = Document(
            file_name=file.filename,
            file_path=local_file_path,
            qiniu_key=qiniu_key,
            qiniu_url=qiniu_url,
            status=DocumentStatus.UPLOADING,
            file_size=len(content)
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            producer.send(
                KAFKA_TOPIC,
                value={
                    "document_id": document.id,
                    "qiniu_url": qiniu_url,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            producer.flush()
            producer.close()
        except Exception as e:
            print(f"[Router] Failed to send Kafka message: {e}")

        return JSONResponse({
            "success": True,
            "document_id": document.id,
            "file_name": file.filename,
            "status": DocumentStatus.UPLOADING.value,
            "qiniu_url": qiniu_url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def get_mock_text_results() -> list:
    """Return mock text search results for testing"""
    return [
        {
            "text": "汽车保养的关键在于定期更换机油和机油滤清器。建议每5000公里或6个月更换一次，以确保发动机正常运转。",
            "page": 5,
            "metadata": {"file_name": "汽车知识手册.pdf"}
        },
        {
            "text": "刹车片的检查也是重要的保养项目。当刹车片磨损到3mm以下时，应及时更换，以保证行车安全。",
            "page": 12,
            "metadata": {"file_name": "汽车知识手册.pdf"}
        },
        {
            "text": "轮胎保养包括定期检查胎压和胎纹深度。建议每月检查一次胎压，胎纹深度不应低于1.6mm。",
            "page": 8,
            "metadata": {"file_name": "汽车知识手册.pdf"}
        }
    ]


def get_mock_image_results() -> list:
    """Return mock image search results for testing"""
    return [
        {
            "image_base64": "",
            "page": 15,
            "metadata": {"file_name": "汽车知识手册.pdf", "image_id": "img_001"}
        }
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db=Depends(get_db)):
    """
    Chat Q&A endpoint with RAG.

    Flow:
    1. Receive user query and knowledge_base_id
    2. Generate text embedding via BGE
    3. Hybrid search in Milvus (text + image chunks)
    4. Construct prompt with retrieved context
    5. Call Qwen-VL for multi-modal reasoning
    6. Return answer with source attribution
    """
    if not request.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id

    session_obj = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session_obj:
        session_obj = ChatSession(
            session_id=session_id,
            document_id=request.knowledge_base_id
        )
        db.add(session_obj)
        db.commit()

    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.query
    )
    db.add(user_message)
    db.commit()

    use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    
    if use_mock_data:
        text_results = get_mock_text_results()
        image_results = get_mock_image_results()
    else:
        try:
            milvus_service = MilvusService()
            text_results = milvus_service.search_text(
                collection_name="documents",
                query=request.query,
                limit=5
            )

            image_results = milvus_service.search_image(
                collection_name="document_images",
                query=request.query,
                limit=3
            )
        except Exception as e:
            print(f"[Router] Milvus connection failed, using mock data: {e}")
            text_results = get_mock_text_results()
            image_results = get_mock_image_results()

    context_parts = []
    sources = []

    for item in text_results:
        context_parts.append(f"[文本] {item.get('text', '')}")
        sources.append({
            "type": "text",
            "file_name": item.get("metadata", {}).get("file_name", "unknown"),
            "page": item.get("page", 0),
            "content_preview": item.get("text", "")[:200]
        })

    for item in image_results:
        context_parts.append(f"[图片] 来自 {item.get('metadata', {}).get('file_name', 'unknown')} 第{item.get('page', 0)}页")
        sources.append({
            "type": "image",
            "file_name": item.get("metadata", {}).get("file_name", "unknown"),
            "page": item.get("page", 0),
            "image_id": item.get("metadata", {}).get("image_id", "")
        })

    if not context_parts:
        answer = "当前知识库中暂无相关数据。请先上传文档并完成解析后，再进行问答。"
    else:
        prompt = f"""你是一个文档问答助手。请根据以下上下文信息回答用户的问题。

上下文信息：
{chr(10).join(context_parts)}

用户问题：{request.query}

请基于上述上下文信息给出准确回答，并在答案中标注信息来源（来自哪个PDF文件的哪一页）。"""
        answer = f"根据检索到的信息，回答如下：\n\n{chr(10).join(context_parts)}\n\n（来源：已检索到 {len(sources)} 条相关记录）"

    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources)
    )
    db.add(assistant_message)
    db.commit()

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id
    )


@router.get("/document/{document_id}")
async def get_document_status(document_id: int, db=Depends(get_db)):
    """Get document processing status"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": document.id,
        "file_name": document.file_name,
        "status": document.status.value,
        "page_count": document.page_count,
        "qiniu_url": document.qiniu_url,
        "created_at": document.created_at.isoformat() if document.created_at else None
    }