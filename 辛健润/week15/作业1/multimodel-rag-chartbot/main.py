"""FastAPI Main Entry."""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import upload, search, chat
from models.orm import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    init_db()
    print("App started!")
    yield
    # Shutdown
    print("App shutdown!")


app = FastAPI(
    title="Multimodal RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(upload.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    uvicorn.run(
        "main:app",
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=True
    )