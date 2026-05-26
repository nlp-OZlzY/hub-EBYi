"""问答接口"""
import logging
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, ErrorResponse
from services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()
chat_service = ChatService()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}},
    summary="智能问答",
    description="用户提问，系统检索相关内容并生成回答"
)
async def chat(request: ChatRequest):
    """
    智能问答接口

    - **question**: 用户问题
    """
    # 1. 参数校验
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 2. 调用Service处理
    try:
        result = chat_service.chat(request.question)
        return ChatResponse(data=result)
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail="问答服务异常")
