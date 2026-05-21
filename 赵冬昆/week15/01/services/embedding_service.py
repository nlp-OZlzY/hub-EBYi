"""Embedding Service using Aliyun DashScope MultiModalEmbedding"""
import os
import dashscope
from dashscope import MultiModalEmbedding

class EmbeddingService:
    """Service class for text embedding using Aliyun DashScope"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-2263f8eb194b4ea89d177ffbe97f2c9d")
        dashscope.api_key = self.api_key
        self.dim = 1152
        print(f"[EmbeddingService] Initialized with DashScope API, dim={self.dim}")

    def embed(self, text: str) -> list:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            list: Embedding vector
        """
        try:
            response = MultiModalEmbedding.call(
                model="tongyi-embedding-vision-plus-2026-03-06",
                input=[{"text": text}]
            )
            if response.status_code == 200:
                return response.output["embeddings"][0]["embedding"]
            else:
                print(f"Embedding API error: {response.message}")
                return []
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            list: List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            if embedding:
                embeddings.append(embedding)
        return embeddings
