"""
多模态RAG聊天机器人 - FastAPI服务
提供文件管理、文档解析、问答等核心API
"""
import os
import uuid
import hashlib
import json
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Path, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orm_model import (
    init_db, KnowledgeBase, File as FileModel, Chunk, ChunkImage,
    ChatSession, ChatMessage, FileState, ChunkType, get_db_path
)
from api_schemas import (
    KnowledgeBaseCreate, KnowledgeBaseResponse, KnowledgeBaseList,
    FileResponse, FileListResponse, FileStatusResponse,
    ChatRequest, ChatResponse, ChatMessage as ChatMessageSchema,
    ChunkSearchRequest, ChunkSearchResponse, ChunkResponse,
    ErrorResponse
)

# ==================== 初始化 ====================
app = FastAPI(
    title="多模态RAG聊天机器人API",
    description="支持图文检索、问答的RAG系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库会话
Session, engine = init_db()

# Milvus客户端配置
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b")
COLLECTION_NAME = "rag_data_new"

# Kafka配置
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "rag-data"

# ==================== 辅助函数 ====================
def get_session():
    """获取数据库会话"""
    return Session()

def calculate_file_hash(file_path: str) -> str:
    """计算文件MD5"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()

def get_milvus_client():
    """获取Milvus客户端"""
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

def send_to_kafka(file_id: int, file_name: str, file_path: str):
    """发送消息到Kafka"""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        producer.send(
            KAFKA_TOPIC,
            value={"file_name": file_name, "file_path": file_path, "id": file_id}
        )
        producer.flush()
        return True
    except Exception as e:
        print(f"Kafka send failed: {e}")
        return False

# ==================== 知识库接口 ====================
@app.post("/api/v1/knowledge-bases", response_model=KnowledgeBaseResponse, tags=["知识库"])
async def create_knowledge_base(req: KnowledgeBaseCreate):
    """创建知识库"""
    session = get_session()
    try:
        # 检查名称是否重复
        existing = session.query(KnowledgeBase).filter(
            KnowledgeBase.name == req.name
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"知识库 '{req.name}' 已存在")

        kb = KnowledgeBase(name=req.name, description=req.description)
        session.add(kb)
        session.commit()
        session.refresh(kb)
        return kb
    finally:
        session.close()

@app.get("/api/v1/knowledge-bases", response_model=KnowledgeBaseList, tags=["知识库"])
async def list_knowledge_bases(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """获取知识库列表"""
    session = get_session()
    try:
        total = session.query(KnowledgeBase).count()
        items = session.query(KnowledgeBase).offset(skip).limit(limit).all()
        return KnowledgeBaseList(items=items, total=total)
    finally:
        session.close()

@app.get("/api/v1/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse, tags=["知识库"])
async def get_knowledge_base(kb_id: int = Path(..., description="知识库ID")):
    """获取知识库详情"""
    session = get_session()
    try:
        kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return kb
    finally:
        session.close()

@app.delete("/api/v1/knowledge-bases/{kb_id}", tags=["知识库"])
async def delete_knowledge_base(kb_id: int = Path(..., description="知识库ID")):
    """删除知识库（同时删除关联的文件和chunks）"""
    session = get_session()
    try:
        kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 删除关联的文件
        files = session.query(FileModel).filter(
            FileModel.knowledge_base_id == kb_id
        ).all()

        for file in files:
            # 删除Milvus中的数据
            try:
                mc = get_milvus_client()
                mc.delete(collection_name=COLLECTION_NAME, filter=f"db_id == {file.id}")
            except:
                pass

            # 删除本地文件
            if os.path.exists(file.filepath):
                os.remove(file.filepath)
            if file.parsed_path and os.path.exists(file.parsed_path):
                os.remove(file.parsed_path)

        session.delete(kb)
        session.commit()
        return {"message": "删除成功"}
    finally:
        session.close()

# ==================== 文件管理接口 ====================
@app.post("/api/v1/files/upload", response_model=FileResponse, tags=["文件管理"])
async def upload_file(
    knowledge_base_id: Optional[int] = Form(None),
    file: UploadFile = File(..., description="上传的文件")
):
    """上传文件到知识库"""
    session = get_session()
    try:
        # 保存文件
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        unique_name = str(uuid.uuid4())
        save_path = os.path.join(upload_dir, unique_name + file_ext)

        # 写入文件
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # 计算文件hash和大小
        file_hash = hashlib.md5(content).hexdigest()
        file_size = len(content)

        # 获取MIME类型
        mime_type = file.content_type or "application/octet-stream"

        # 创建记录
        file_record = FileModel(
            filename=file.filename,
            filepath=save_path,
            file_size=file_size,
            file_hash=file_hash,
            mime_type=mime_type,
            state=FileState.UPLOADED.value,
            knowledge_base_id=knowledge_base_id
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)

        # 发送到Kafka等待处理
        send_to_kafka(file_record.id, file.filename, save_path)

        return file_record
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get("/api/v1/files", response_model=FileListResponse, tags=["文件管理"])
async def list_files(
    knowledge_base_id: Optional[int] = Query(None),
    state: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """获取文件列表"""
    session = get_session()
    try:
        query = session.query(FileModel)
        if knowledge_base_id is not None:
            query = query.filter(FileModel.knowledge_base_id == knowledge_base_id)
        if state is not None:
            query = query.filter(FileModel.state == state)

        total = query.count()
        items = query.order_by(FileModel.created_at.desc()).offset(skip).limit(limit).all()
        return FileListResponse(items=items, total=total)
    finally:
        session.close()

@app.get("/api/v1/files/{file_id}", response_model=FileResponse, tags=["文件管理"])
async def get_file(file_id: int = Path(..., description="文件ID")):
    """获取文件详情"""
    session = get_session()
    try:
        file = session.query(FileModel).filter(FileModel.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")
        return file
    finally:
        session.close()

@app.delete("/api/v1/files/{file_id}", tags=["文件管理"])
async def delete_file(file_id: int = Path(..., description="文件ID")):
    """删除文件"""
    session = get_session()
    try:
        file = session.query(FileModel).filter(FileModel.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")

        # 删除Milvus数据
        try:
            mc = get_milvus_client()
            mc.delete(collection_name=COLLECTION_NAME, filter=f"db_id == {file_id}")
        except:
            pass

        # 删除本地文件
        if os.path.exists(file.filepath):
            os.remove(file.filepath)
        if file.parsed_path and os.path.exists(file.parsed_path):
            os.remove(file.parsed_path)

        session.delete(file)
        session.commit()
        return {"message": "删除成功"}
    finally:
        session.close()

@app.get("/api/v1/files/{file_id}/status", response_model=FileStatusResponse, tags=["文件管理"])
async def get_file_status(file_id: int = Path(..., description="文件ID")):
    """获取文件处理状态"""
    session = get_session()
    try:
        file = session.query(FileModel).filter(FileModel.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")

        # 获取chunk数量
        chunk_count = session.query(Chunk).filter(Chunk.file_id == file_id).count()

        # 计算进度
        progress = None
        if file.state == FileState.UPLOADED.value:
            progress = 0.0
        elif file.state == FileState.PARSING.value:
            progress = 0.3
        elif file.state == FileState.PARSED.value:
            progress = 0.5
        elif file.state == FileState.INDEXING.value:
            progress = 0.7
        elif file.state == FileState.INDEXED.value:
            progress = 1.0

        return FileStatusResponse(
            file_id=file_id,
            state=file.state,
            state_message=file.state_message,
            progress=progress,
            parsed_path=file.parsed_path,
            page_count=file.page_count,
            chunk_count=chunk_count
        )
    finally:
        session.close()

# ==================== 文档解析接口 ====================
@app.post("/api/v1/files/{file_id}/parse", tags=["文档解析"])
async def parse_document(file_id: int = Path(..., description="文件ID")):
    """手动触发文档解析"""
    session = get_session()
    try:
        file = session.query(FileModel).filter(FileModel.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")

        if file.state not in [FileState.UPLOADED.value, FileState.PARSE_FAILED.value]:
            raise HTTPException(status_code=400, detail=f"文件状态为 {file.state}，无法解析")

        # 更新状态为解析中
        file.state = FileState.PARSING.value
        session.commit()

        # 发送到Kafka
        send_to_kafka(file.id, file.filename, file.filepath)

        return {"message": "解析任务已提交", "file_id": file_id}
    finally:
        session.close()

# ==================== Chunk搜索接口 ====================
@app.post("/api/v1/chunks/search", response_model=ChunkSearchResponse, tags=["检索"])
async def search_chunks(req: ChunkSearchRequest):
    """搜索相关的Chunks"""
    session = get_session()
    try:
        # 获取该知识库下的所有文件ID
        files = session.query(FileModel.id).filter(
            FileModel.knowledge_base_id == req.knowledge_base_id,
            FileModel.state == FileState.INDEXED.value
        ).all()
        file_ids = [f.id for f in files]

        if not file_ids:
            return ChunkSearchResponse(items=[], total=0, query=req.query)

        # 搜索Milvus
        mc = get_milvus_client()

        # 对query进行embedding
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        from sentence_transformers import SentenceTransformer
        bge_model = SentenceTransformer('/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5')
        query_embedding = bge_model.encode(req.query, normalize_embeddings=True)

        # 执行搜索
        results = mc.search(
            collection_name=COLLECTION_NAME,
            data=[list(query_embedding)],
            filter=f"db_id in {[fid for fid in file_ids]}",
            limit=req.limit,
            output_fields=["text", "db_id", "file_name", "file_path"]
        )

        # 转换结果
        items = []
        for result in results[0]:
            entity = result["entity"]
            # 获取关联的图片
            chunk_images = session.query(ChunkImage).join(Chunk).filter(
                Chunk.file_id == entity["db_id"],
                Chunk.content.contains(entity["text"][:100])  # 简单匹配
            ).limit(5).all()

            items.append(ChunkResponse(
                id=result["id"],
                content=entity["text"],
                content_type=ChunkType.TEXT.value,
                page_num=None,
                chunk_index=0,
                file_id=entity["db_id"],
                images=[img.image_path for img in chunk_images]
            ))

        return ChunkSearchResponse(items=items, total=len(items), query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get("/api/v1/files/{file_id}/chunks", response_model=List[ChunkResponse], tags=["检索"])
async def get_file_chunks(
    file_id: int = Path(..., description="文件ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """获取某个文件的chunks"""
    session = get_session()
    try:
        chunks = session.query(Chunk).filter(
            Chunk.file_id == file_id
        ).order_by(Chunk.chunk_index).offset(skip).limit(limit).all()

        result = []
        for chunk in chunks:
            images = session.query(ChunkImage).filter(
                ChunkImage.chunk_id == chunk.id
            ).all()
            result.append(ChunkResponse(
                id=chunk.id,
                content=chunk.content,
                content_type=chunk.content_type,
                page_num=chunk.page_num,
                chunk_index=chunk.chunk_index,
                file_id=chunk.file_id,
                images=[img.image_path for img in images]
            ))
        return result
    finally:
        session.close()

# ==================== 问答接口 ====================
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["问答"])
async def chat(req: ChatRequest):
    """进行问答"""
    session = get_session()
    try:
        # 获取或创建会话
        session_id = req.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_session = ChatSession(
                id=session_id,
                knowledge_base_id=req.knowledge_base_id
            )
            session.add(chat_session)
        else:
            chat_session = session.query(ChatSession).filter(
                ChatSession.id == session_id
            ).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="会话不存在")

        # 保存用户消息
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=req.query
        )
        session.add(user_msg)
        session.commit()

        # 检索相关chunks
        files = session.query(FileModel.id).filter(
            FileModel.knowledge_base_id == req.knowledge_base_id,
            FileModel.state == FileState.INDEXED.value
        ).all()
        file_ids = [f.id for f in files]

        retrieved_chunks = []
        if file_ids:
            mc = get_milvus_client()

            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            from sentence_transformers import SentenceTransformer
            bge_model = SentenceTransformer('/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5')
            query_embedding = bge_model.encode(req.query, normalize_embeddings=True)

            results = mc.search(
                collection_name=COLLECTION_NAME,
                data=[list(query_embedding)],
                filter=f"db_id in {[fid for fid in file_ids]}",
                limit=req.top_k,
                output_fields=["text", "db_id", "file_name", "file_path"]
            )

            for result in results[0]:
                retrieved_chunks.append({
                    "id": result["id"],
                    "text": result["entity"]["text"],
                    "file_name": result["entity"]["file_name"],
                    "file_path": result["entity"]["file_path"],
                    "score": result["distance"]
                })

        # 构建prompt
        related_content = "\n".join([
            f"[来源: {c['file_name']}]\n{c['text']}" for c in retrieved_chunks
        ])

        rag_prompt = f"""基于资料回答的提问提问问题：{req.query}

相关资料: {related_content}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接。
- 指出答案的信息来源。"""

        # 调用大模型
        from openai import OpenAI
        qwen_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", "sk-711c186f74494136ba26035be25a7cb8"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': rag_prompt}
            ],
        )

        answer = completion.choices[0].message.content

        # 保存助手消息
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            source_chunks=json.dumps([c["id"] for c in retrieved_chunks])
        )
        session.add(assistant_msg)

        # 更新会话统计
        chat_session.message_count += 2
        session.commit()

        return ChatResponse(
            session_id=session_id,
            message=ChatMessageSchema(
                role="assistant",
                content=answer,
                source_chunks=[c["id"] for c in retrieved_chunks]
            ),
            retrieved_chunks=retrieved_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get("/api/v1/chat/sessions/{session_id}/history", tags=["问答"])
async def get_chat_history(
    session_id: str = Path(..., description="会话ID"),
    limit: int = Query(50, ge=1, le=200)
):
    """获取聊天历史"""
    session = get_session()
    try:
        chat_session = session.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="会话不存在")

        messages = session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).limit(limit).all()

        return {
            "session_id": session_id,
            "messages": [
                ChatMessageSchema(
                    role=msg.role,
                    content=msg.content,
                    source_chunks=json.loads(msg.source_chunks) if msg.source_chunks else None
                ) for msg in messages
            ],
            "total": len(messages)
        }
    finally:
        session.close()

# ==================== 健康检查 ====================
@app.get("/api/v1/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_path": get_db_path(),
        "milvus_uri": MILVUS_URI[:20] + "..." if MILVUS_URI else None
    }

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)