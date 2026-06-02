"""
EmbeddingsCache.py

功能：
    把文本对应的 embedding 向量缓存到 Redis 中，避免同一段文本重复调用
    Embedding 模型。

核心设计：
    1. 文本先做 MD5 哈希，作为 Redis key 的一部分，避免中文、空格、标点
       直接进入 key 后带来的兼容性问题。
    2. numpy 向量统一转成 float32，再用 tobytes() 存入 Redis。
    3. 查询时用 np.frombuffer(..., dtype=np.float32) 把 bytes 还原为向量。
    4. 支持单条文本和批量文本，内部统一转换为 list 处理。

Redis 数据结构：
    Key:   {name}:{md5(text)}
    Value: embedding 的 float32 bytes
    TTL:   缓存过期时间，默认 24 小时
"""

import hashlib
from typing import List, Union

import numpy as np
import redis


TextInput = Union[str, List[str]]


class EmbeddingsCache:
    def __init__(
        self,
        name: str,
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
    ):
        self.name = name
        self.ttl = ttl
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

    def _make_key(self, text: str) -> str:
        """根据文本生成稳定的 Redis key。"""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self.name}:{text_hash}"

    @staticmethod
    def _normalize_text(text: TextInput) -> List[str]:
        """把单条文本和批量文本统一成 list[str]。"""
        if isinstance(text, str):
            return [text]
        return list(text)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray, expected_rows: int) -> np.ndarray:
        """
        把 embedding 统一成二维 float32 数组。

        单条文本常见输入形状是 (768,)，批量文本常见输入形状是 (N, 768)。
        Redis 中只保存 float32 bytes，所以这里提前统一 dtype。
        """
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if embedding.ndim != 2:
            raise ValueError("embedding 必须是一维或二维 numpy 数组")

        if embedding.shape[0] != expected_rows:
            raise ValueError(
                f"text 数量是 {expected_rows}，但 embedding 行数是 {embedding.shape[0]}"
            )

        return np.ascontiguousarray(embedding)

    def store(self, text: TextInput, embedding: np.ndarray):
        """
        存入文本和对应 embedding。

        参数：
            text: 单条文本或文本列表。
            embedding: 单条向量或向量矩阵，必须与 text 一一对应。

        返回：
            Redis pipeline 的执行结果；失败时返回 -1。
        """
        texts = self._normalize_text(text)
        if not texts:
            return []

        try:
            embeddings = self._normalize_embedding(embedding, len(texts))

            with self.redis.pipeline() as pipe:
                for text_item, vector in zip(texts, embeddings):
                    # 每个文本独立设置 TTL，避免缓存永久堆积。
                    pipe.setex(
                        self._make_key(text_item),
                        self.ttl,
                        vector.astype(np.float32).tobytes(),
                    )
                return pipe.execute()
        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, text: TextInput):
        """
        读取缓存中的 embedding。

        返回：
            list[np.ndarray | None]
            命中的位置返回向量，未命中的位置返回 None。即使传入单条文本，
            也返回列表，方便和批量调用保持一致。
        """
        texts = self._normalize_text(text)
        if not texts:
            return []

        try:
            keys = [self._make_key(text_item) for text_item in texts]
            results = self.redis.mget(keys)

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    # copy() 让返回数组脱离 Redis bytes buffer，后续可安全修改。
                    embeddings.append(np.frombuffer(result, dtype=np.float32).copy())
            return embeddings
        except Exception as e:
            print(f"Call error: {e}")
            return None

    def delete(self, text: TextInput):
        """
        删除指定文本的 embedding 缓存。

        返回：
            Redis 实际删除的 key 数量；失败时返回 -1。
        """
        texts = self._normalize_text(text)
        if not texts:
            return 0

        try:
            keys = [self._make_key(text_item) for text_item in texts]
            return self.redis.delete(*keys)
        except Exception as e:
            print(f"Delete error: {e}")
            return -1


if __name__ == "__main__":
    cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="localhost",
    )

    def get_embedding(text: str) -> np.ndarray:
        """模拟 Embedding 模型：实际项目中可替换为 OpenAI 或本地模型。"""
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(768, dtype=np.float32)

    print("=== store ===")
    print(cache.store("hello world", get_embedding("hello world")))

    print("=== call ===")
    result = cache.call("hello world")
    vector_shape = result[0].shape if result and result[0] is not None else None
    print(f"返回类型: {type(result)}, 向量维度: {vector_shape}")

    print("=== delete ===")
    print(cache.delete("hello world"))

    print("=== call after delete ===")
    print(cache.call("hello world"))
