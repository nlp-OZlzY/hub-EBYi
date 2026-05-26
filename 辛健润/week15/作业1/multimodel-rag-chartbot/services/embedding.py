"""Embedding Service - BGE + CLIP."""
import os
import yaml
import numpy as np
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from sentence_transformers import SentenceTransformer

_config = None


def load_config():
    global _config
    if _config is None:
        cfg_path = "config.yaml"
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(cfg_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


class EmbeddingService:
    def __init__(self):
        cfg = load_config()
        emb_cfg = cfg["embedding"]

        print("Loading BGE model...")
        self.bge_model = SentenceTransformer(emb_cfg["bge_model"])

        print("Loading CLIP model...")
        self.clip_model = SentenceTransformer(
            emb_cfg["clip_model"],
            trust_remote_code=True,
            truncate_dim=1024
        )

    def encode_text(self, text: str, normalize: bool = True):
        """BGE文本向量化"""
        emb = self.bge_model.encode(text, normalize_embeddings=normalize)
        return list(emb)

    def encode_image(self, image_path: str, normalize: bool = True):
        """CLIP图像向量化"""
        try:
            emb = self.clip_model.encode(image_path, normalize_embeddings=normalize)
            return list(emb)
        except Exception:
            traceback.print_exc()
            return list(np.zeros(1024))

    def encode_text_for_clip(self, text: str, normalize: bool = True):
        """CLIP文本向量化"""
        try:
            emb = self.clip_model.encode(text, normalize_embeddings=normalize)
            return list(emb)
        except Exception:
            traceback.print_exc()
            return list(np.zeros(1024))


# Global singleton
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service