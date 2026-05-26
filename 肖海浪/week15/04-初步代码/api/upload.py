"""上传接口"""
import os
import logging
from fastapi import APIRouter, UploadFile, HTTPException
from models.schemas import UploadResponse, ErrorResponse
from services.file_service import FileService
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
file_service = FileService()


@router.post(
    "/upload/document",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    summary="上传文档",
    description="上传PDF/DOCX/TXT文件到指定知识库"
)
async def upload_document(file: UploadFile):
    """
    上传文档接口

    - **file**: 上传的文件（支持 .pdf / .docx / .txt）
    """
    # 1. 校验文件类型
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {'/'.join(settings.ALLOWED_EXTENSIONS)} 格式"
        )

    # 2. 读取文件内容
    content = await file.read()

    # 3. 校验文件大小
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件不能为空")

    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制")

    # 4. 调用Service处理
    try:
        result = file_service.save_and_publish(file.filename, content)
        return UploadResponse(data=result)
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")
