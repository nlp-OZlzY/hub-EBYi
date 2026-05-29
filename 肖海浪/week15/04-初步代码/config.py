"""配置文件"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 文件存储
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".txt"]

    # SQLite
    DB_PATH: str = "db.db"

    # Milvus
    MILVUS_URI: str = ""
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION: str = "rag_data_new"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC: str = "rag-data"

    # 模型路径
    BGE_MODEL_PATH: str = "/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5"
    CLIP_MODEL_PATH: str = "/root/autodl-tmp/models/jinaai/jina-clip-v2"

    # LLM
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "deepseek-v4-pro"

    # 文本切分
    CHUNK_SIZE: int = 256

    class Config:
        env_file = ".env"


settings = Settings()
