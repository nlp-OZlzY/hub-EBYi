"""文件管理接口"""
import logging
from fastapi import APIRouter, HTTPException
from models.schemas import FileListResponse, DeleteResponse, ErrorResponse
from services.file_service import FileService
from services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()
file_service = FileService()
search_service = SearchService()


@router.get(
    "/files",
    response_model=FileListResponse,
    summary="文件列表",
    description="查询已上传的文件列表"
)
async def list_files():
    """获取所有文件列表"""
    files = file_service.get_all_files()
    return FileListResponse(data=files)


@router.delete(
    "/files/{file_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="删除文件",
    description="删除指定文件及其向量数据"
)
async def delete_file(file_id: int):
    """
    删除文件

    - **file_id**: 文件ID
    """
    # 1. 检查文件是否存在
    file = file_service.get_file_by_id(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")

    # 2. 删除本地文件和数据库记录
    file_service.delete_file(file_id)

    # 3. 清理Milvus向量数据
    try:
        search_service.delete_by_file_id(file_id)
    except Exception as e:
        logger.error(f"Milvus清理失败: {e}")

    return DeleteResponse()
