"""启动 API 服务（含前端静态资源，需先 build frontend）"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
