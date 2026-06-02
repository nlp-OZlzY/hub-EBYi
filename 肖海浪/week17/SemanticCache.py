"""
SemanticCache.py

功能：
    缓存 LLM 的问答结果。后续问题如果和已缓存问题语义相近，就直接返回
    缓存回答，减少重复调用大模型的成本。

实现思路：
    - FAISS 保存 prompt 的 embedding，并负责最近邻检索。
    - Redis 保存 prompt 到 response 的映射，以及 prompt 顺序列表。
    - FAISS 索引位置必须和 Redis prompt 列表位置保持一致，才能通过
      向量检索结果反查原始问题。

本作业与 redis-vl-python 的区别：
    redis-vl-python 使用 Redis Stack/RediSearch 原生向量索引；
    这里为了便于学习，使用 FAISS 做本地向量索引，Redis 只负责 KV 存储。
"""

import hashlib
import os
from typing import Any, Callable, List, Union

import faiss
import numpy as np
import redis


TextInput = Union[str, List[str]]


class SemanticCache:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[TextInput], Any],
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold: float = 0.1,
    ):
        self.name = name
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.index_path = f"{self.name}.index"
        self.prompts_key = f"{self.name}:prompts"

        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

        # 如果本地已有 FAISS 索引文件，就尝试恢复。
        self.index = faiss.read_index(self.index_path) if os.path.exists(self.index_path) else None

    def _response_key(self, prompt: str) -> str:
        """prompt 可能很长，先哈希再作为 Redis key 的一部分。"""
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return f"{self.name}:response:{prompt_hash}"

    @staticmethod
    def _as_list(value: TextInput) -> List[str]:
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _as_2d_float32(embedding: Any) -> np.ndarray:
        """FAISS 要求输入是二维、连续内存、float32 的 numpy 数组。"""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.ndim != 2:
            raise ValueError("embedding_method 必须返回一维或二维向量")
        return np.ascontiguousarray(embedding)

    def store(self, prompt: TextInput, response: TextInput):
        """
        缓存 prompt 和 response。

        参数：
            prompt: 单条问题或问题列表。
            response: 单条回答或回答列表，数量必须和 prompt 对齐。

        返回：
            Redis pipeline 执行结果；失败时返回 -1。
        """
        prompts = self._as_list(prompt)
        responses = self._as_list(response)

        if len(prompts) != len(responses):
            raise ValueError("prompt 和 response 的数量必须一致")
        if not prompts:
            return []

        embeddings = self._as_2d_float32(self.embedding_method(prompts))
        if embeddings.shape[0] != len(prompts):
            raise ValueError("embedding 数量必须和 prompt 数量一致")

        # 第一次写入时根据 embedding 维度创建 FAISS 索引。
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        if self.index.d != embeddings.shape[1]:
            raise ValueError(f"embedding 维度应为 {self.index.d}，实际为 {embeddings.shape[1]}")

        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        try:
            with self.redis.pipeline() as pipe:
                for prompt_item, response_item in zip(prompts, responses):
                    # rpush 保证 Redis prompt 顺序和 FAISS add 顺序一致。
                    pipe.rpush(self.prompts_key, prompt_item)
                    pipe.setex(self._response_key(prompt_item), self.ttl, response_item)

                # prompt 列表本身也设置 TTL，和回答缓存生命周期保持一致。
                pipe.expire(self.prompts_key, self.ttl)
                return pipe.execute()
        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, prompt: str, top_k: int = 5):
        """
        查询语义相近的缓存回答。

        返回：
            命中时返回 Redis mget 的 bytes 列表，例如 [b'answer']；
            未命中时返回 None。
        """
        if self.index is None or self.index.ntotal == 0:
            return None

        embedding = self._as_2d_float32(self.embedding_method(prompt))
        k = min(max(top_k, 1), self.index.ntotal)
        distances, indices = self.index.search(embedding, k=k)

        raw_prompts = self.redis.lrange(self.prompts_key, 0, -1)
        if not raw_prompts:
            return None

        matched_prompts = []
        for distance, index_id in zip(distances[0], indices[0]):
            # FAISS 在结果不足时可能返回 -1；距离超过阈值也视为未命中。
            if index_id < 0 or distance > self.distance_threshold:
                continue
            if index_id >= len(raw_prompts):
                continue
            matched_prompts.append(raw_prompts[index_id].decode("utf-8"))

        if not matched_prompts:
            return None

        keys = [self._response_key(prompt_item) for prompt_item in matched_prompts]
        answers = self.redis.mget(keys)

        # Redis key 可能因 TTL 过期而失效，过滤掉 None。
        answers = [answer for answer in answers if answer is not None]
        return answers or None

    def clear_cache(self):
        """清空 Redis 中的问答缓存和本地 FAISS 索引文件。"""
        raw_prompts = self.redis.lrange(self.prompts_key, 0, -1)

        if raw_prompts:
            response_keys = [
                self._response_key(prompt.decode("utf-8"))
                for prompt in raw_prompts
            ]
            self.redis.delete(*response_keys)

        self.redis.delete(self.prompts_key)

        if os.path.exists(self.index_path):
            os.unlink(self.index_path)

        self.index = None


if __name__ == "__main__":
    def get_embedding(text: TextInput) -> np.ndarray:
        """
        可复现的演示 embedding。

        实际业务里应替换成真正的 embedding 模型。这里按关键词映射向量，
        让"hello world"和"hello there"、"如何退货"和"我要退货"能被视为相近。
        """
        texts = [text] if isinstance(text, str) else list(text)
        vectors = []

        for item in texts:
            lower = item.lower()
            vector = np.zeros(8, dtype=np.float32)
            if "hello" in lower or "hi" in lower or "你好" in item:
                vector[0] = 1.0
            if "退货" in item or "退款" in item or "refund" in lower or "return" in lower:
                vector[1] = 1.0
            if "天气" in item or "weather" in lower:
                vector[2] = 1.0
            vector[7] = 1.0
            vectors.append(vector)

        return np.array(vectors, dtype=np.float32)

    cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=0.01,
    )

    cache.clear_cache()

    print("=== store greeting ===")
    print(cache.store("hello world", "hello answer"))

    print("=== call greeting ===")
    print(cache.call("hello there"))

    print("=== store refund ===")
    print(cache.store("如何退货", "请在订单页面申请退货"))

    print("=== call refund ===")
    print(cache.call("我要退货"))

    print("=== clear ===")
    cache.clear_cache()

    print("=== call after clear ===")
    print(cache.call("hello world"))
