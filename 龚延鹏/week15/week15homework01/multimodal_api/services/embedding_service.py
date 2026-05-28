"""
Embedding 服务（占位）
"""
from typing import List


class EmbeddingService:
    """Embedding 服务（占位）"""

    def encode_text(self, text: str) -> List[float]:
        """文本 embedding（占位）"""
        # TODO: 加载 BGE 模型并编码
        import numpy as np
        return np.random.rand(512).tolist()

    def encode_image(self, image_path: str) -> List[float]:
        """图片 embedding（占位）"""
        # TODO: 加载 CLIP 模型并编码
        import numpy as np
        return np.random.rand(1024).tolist()


embedding_service = EmbeddingService()