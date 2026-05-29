"""Upload API Routes."""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from models.orm import create_file_record, get_file as orm_get_file, list_files, update_file_state, FileState

router = APIRouter(prefix="/files", tags=["files"])


class FileResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    state: str


class FileDetailResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    state: str
    created_at: str
    updated_at: str


def _get_state_str(state: FileState) -> str:
    return state.value if isinstance(state, FileState) else str(state)


@router.get("", response_model=List[FileResponse])
async def list_all_files():
    """列出所有文件"""
    files = list_files()
    return [
        FileResponse(
            id=f.id,
            filename=f.filename,
            filepath=f.filepath,
            state=_get_state_str(f.filestate)
        )
        for f in files
    ]


@router.get("/{file_id}", response_model=FileDetailResponse)
async def get_file_detail(file_id: int):
    """获取文件详情"""
    f = orm_get_file(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    return FileDetailResponse(
        id=f.id,
        filename=f.filename,
        filepath=f.filepath,
        state=_get_state_str(f.filestate),
        created_at=f.created_at.isoformat(),
        updated_at=f.updated_at.isoformat()
    )


@router.post("/upload", response_model=FileResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    # 获取配置中的上传目录
    from services.embedding import load_config
    cfg = load_config()
    uploads_dir = cfg.get("paths", {}).get("uploads_dir", "uploads")

    # 创建上传目录
    os.makedirs(uploads_dir, exist_ok=True)

    # 生成唯一文件名
    ext = os.path.splitext(file.filename)[1]
    unique_name = str(uuid.uuid4()) + ext
    save_path = os.path.join(uploads_dir, unique_name)

    # 保存文件
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 创建DB记录
    record = create_file_record(file.filename, save_path)

    return FileResponse(
        id=record.id,
        filename=record.filename,
        filepath=record.filepath,
        state=_get_state_str(record.filestate)
    )