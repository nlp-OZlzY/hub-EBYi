"""
FastAPI 应用入口

这是整个项目的启动文件，负责：
1. 创建FastAPI应用实例
2. 配置CORS（跨域资源共享）
3. 注册各个模块的路由
4. 启动HTTP服务器

运行方式：python main.py
访问文档：http://localhost:8000/docs
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import upload, chat, files

# ============================================
# 日志配置
# 作用：控制台输出日志，方便调试
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# 创建必要的目录
# exist_ok=True 表示目录已存在时不报错
# ============================================
os.makedirs("uploads", exist_ok=True)    # 上传文件存储目录
os.makedirs("processed", exist_ok=True)  # 解析后的文件存储目录

# ============================================
# 创建FastAPI应用实例
# - title: API文档标题
# - description: API文档描述
# - version: API版本号
# ============================================
app = FastAPI(
    title="多模态RAG聊天机器人",
    description="支持图文混排PDF文档的上传、解析、检索和问答",
    version="1.0.0"
)

# ============================================
# CORS中间件配置
# 作用：允许前端跨域访问后端API
# allow_origins=["*"] 表示允许所有来源
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 注册路由
# 把api/目录下的接口注册到应用中
# tags用于API文档分组
# ============================================
app.include_router(upload.router, tags=["文件上传"])   # 上传相关接口
app.include_router(chat.router, tags=["智能问答"])     # 问答相关接口
app.include_router(files.router, tags=["文件管理"])    # 文件管理接口


@app.get("/", summary="健康检查")
async def root():
    """
    健康检查接口

    用于检查服务是否正常运行
    访问：GET http://localhost:8000/
    """
    return {"status": "ok", "message": "多模态RAG聊天机器人服务运行中"}


# ============================================
# 程序入口
# 当直接运行 python main.py 时执行
# ============================================
if __name__ == "__main__":
    import uvicorn
    # 启动HTTP服务器
    # host="0.0.0.0" 表示监听所有网络接口
    # port=8000 表示监听8000端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
