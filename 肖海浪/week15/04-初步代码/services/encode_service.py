"""向量编码服务"""
import os
import logging
import numpy as np
from typing import Tuple
from config import settings

logger = logging.getLogger(__name__)


class EncodeService:
    """向量编码服务"""

    def __init__(self):
        self.bge_model = None
        self.clip_model = None

    def load_models(self):
        """加载模型"""
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        from sentence_transformers import SentenceTransformer

        self.bge_model = SentenceTransformer(settings.BGE_MODEL_PATH)
        logger.info("bge模型加载完成")

        self.clip_model = SentenceTransformer(
            settings.CLIP_MODEL_PATH,
            trust_remote_code=True,
            truncate_dim=1024
        )
        logger.info("clip模型加载完成")

    def encode_text(self, text: str) -> list:
        """
        bge编码文本

        Args:
            text: 输入文本

        Returns:
            512维向量
        """
        if self.bge_model is None:
            self.load_models()

        try:
            embedding = self.bge_model.encode(text, normalize_embeddings=True)
            return list(embedding)
        except Exception as e:
            logger.error(f"bge编码失败: {e}")
            return [0.0] * 512

    def encode_text_clip(self, text: str) -> list:
        """
        clip编码文本

        Args:
            text: 输入文本

        Returns:
            1024维向量
        """
        if self.clip_model is None:
            self.load_models()

        try:
            embedding = self.clip_model.encode(text, normalize_embeddings=True)
            return list(embedding)
        except Exception as e:
            logger.error(f"clip文本编码失败: {e}")
            return [0.0] * 1024

    def encode_image(self, image_path: str) -> list:
        """
        clip编码图片

        Args:
            image_path: 图片路径

        Returns:
            1024维向量
        """
        if self.clip_model is None:
            self.load_models()

        try:
            embedding = self.clip_model.encode(image_path, normalize_embeddings=True)
            return list(embedding)
        except Exception as e:
            logger.error(f"clip图片编码失败: {e}")
            return [0.0] * 1024

    def encode_text_and_image(self, text: str, markdown_path: str) -> Tuple[list, list, list]:
        """
        编码文本和图片

        Args:
            text: 文本内容
            markdown_path: markdown文件路径

        Returns:
            (text_bge_embedding, text_clip_embedding, image_clip_embedding)
        """
        # 分离文本和图片引用
        text_with_no_image = "\n".join([line for line in text.split("\n") if not line.startswith("![")])
        text_with_image = [line for line in text.split("\n") if line.startswith("![")]

        # bge编码文本
        try:
            text_bge_embedding = self.encode_text(text_with_no_image)
        except Exception as e:
            logger.error(f"bge编码异常: {e}")
            text_bge_embedding = [0.0] * 512

        # clip编码文本
        try:
            text_clip_embedding = self.encode_text_clip(text_with_no_image)
        except Exception as e:
            logger.error(f"clip文本编码异常: {e}")
            text_clip_embedding = [0.0] * 1024

        # clip编码图片
        if len(text_with_image) > 0:
            image_path = text_with_image[0].split("](")[1].split(")")[0]
            image_real_path = os.path.dirname(markdown_path) + "/" + image_path.split("/")[-1]
            try:
                logger.info(f"编码图片: {image_real_path}")
                image_clip_embedding = self.encode_image(image_real_path)
            except Exception as e:
                logger.error(f"clip图片编码异常: {e}")
                image_clip_embedding = [0.0] * 1024
        else:
            image_clip_embedding = [0.0] * 1024

        return text_bge_embedding, text_clip_embedding, image_clip_embedding
