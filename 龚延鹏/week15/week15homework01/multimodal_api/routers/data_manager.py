"""
数据管理接口
- 数据上传（文本/图片/文件基础占位）
- 数据删除
- 数据列表查询
"""
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional, List
import uuid
import os
from core.response import ResponseModel

router = APIRouter()

# 模拟数据存储（内存持久化）
DATA_STORE = []


@router.post("/data/upload")
async def upload_data(
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None)
):
    """数据上传接口（占位）"""
    file_id = len(DATA_STORE) + 1

    if file:
        # 保存文件到本地
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
        save_name = f"{uuid.uuid4()}{file_ext}"
        save_dir = "./uploads"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        item = {
            "id": file_id,
            "filename": file.filename,
            "filepath": save_path,
            "type": "file",
            "content": None
        }
    else:
        item = {
            "id": file_id,
            "filename": None,
            "filepath": None,
            "type": "text",
            "content": content
        }

    DATA_STORE.append(item)
    return ResponseModel.success(data=item, msg="上传成功")


@router.post("/data/delete")
async def delete_data(ids: List[int]):
    """数据删除接口"""
    global DATA_STORE
    original_len = len(DATA_STORE)
    DATA_STORE = [item for item in DATA_STORE if item["id"] not in ids]
    deleted_count = original_len - len(DATA_STORE)
    return ResponseModel.success(data={"deleted": deleted_count}, msg="删除成功")


@router.get("/data/list")
async def list_data(limit: int = 10, offset: int = 0):
    """数据列表查询接口"""
    total = len(DATA_STORE)
    items = DATA_STORE[offset:offset + limit] if DATA_STORE else []
    return ResponseModel.success(data={
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items
    }, msg="查询成功")