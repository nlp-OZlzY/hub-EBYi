"""
统一响应工具
"""
from typing import Any, Optional
from fastapi.responses import JSONResponse


class ResponseModel:
    """统一响应格式封装"""

    @staticmethod
    def success(data: Any = None, msg: str = "success") -> JSONResponse:
        return JSONResponse(
            content={
                "code": 200,
                "msg": msg,
                "data": data
            }
        )

    @staticmethod
    def error(code: int = 500, msg: str = "error", data: Any = None) -> JSONResponse:
        return JSONResponse(
            content={
                "code": code,
                "msg": msg,
                "data": data
            }
        )