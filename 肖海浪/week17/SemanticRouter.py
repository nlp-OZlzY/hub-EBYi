"""
SemanticRouter.py

功能：
    根据用户输入的语义，把请求分发到最匹配的业务路由。

例子：
    "你好呀"       -> greeting
    "我要退货"     -> refund
    "帮我查物流"   -> logistics

设计思路：
    1. add_route 阶段：为每个 route 准备若干示例问题，并生成 embedding。
    2. 检索阶段：把用户输入也生成 embedding，在 FAISS 中找最近的示例问题。
    3. 通过最近邻的索引位置，找到对应的 route 名称并返回。

数据结构：
    FAISS:
        保存所有示例问题的 embedding。
    Redis:
        {name}:route_names  -> route 名称列表，顺序与 FAISS 索引一致
        {name}:examples     -> 示例问题列表，便于调试和恢复
        {name}:emb:{i}      -> 第 i 个示例问题的 embedding bytes

说明：
    redis-vl-python 的 SemanticRouter 通常依赖 Redis Stack 的向量索引。
    本作业使用 FAISS + Redis 手写一个教学版，便于看清楚底层流程。
"""

import json
import re
from typing import Any, Callable, List, Optional, Union

import faiss
import numpy as np
import redis


TextInput = Union[str, List[str]]


class SemanticRouter:
    def __init__(
        self,
        name: str = "semantic_router",
        embedding_method: Optional[Callable[[TextInput], Any]] = None,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold: Optional[float] = None,
        ttl: Optional[int] = None,
    ):
        self.name = name
        self.embedding_method = embedding_method or self._default_embedding
        self.distance_threshold = distance_threshold
        self.ttl = ttl

        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

        self.index = None
        self._dim = None
        self._route_names: List[str] = []
        self._examples: List[str] = []

        # 如果 Redis 中保存过路由配置，启动时自动恢复 FAISS 索引。
        self._load_from_redis()

    @staticmethod
    def _default_embedding(text: TextInput) -> np.ndarray:
        """
        仅用于演示的简易 embedding。

        真实项目中应传入 OpenAI、SentenceTransformer 等模型的 embedding
        函数。这里按关键词映射，让示例可以直接运行。
        """
        texts = [text] if isinstance(text, str) else list(text)
        vectors = []

        for item in texts:
            lower = item.lower()
            vector = np.zeros(8, dtype=np.float32)

            if re.search(r"\b(hi|hello|hey)\b", lower) or any(
                word in item for word in ["你好", "您好", "早上好", "下午好", "嗨"]
            ):
                vector[0] = 1.0
            elif any(word in item for word in ["退货", "退款", "退钱"]) or re.search(
                r"\b(refund|return)\b", lower
            ):
                vector[1] = 1.0
            elif any(word in item for word in ["物流", "快递", "到哪"]) or re.search(
                r"\b(logistics|delivery|shipping)\b", lower
            ):
                vector[2] = 1.0
            else:
                # 未命中关键词时，用字符哈希做一个稳定兜底向量。
                for char in item[:64]:
                    vector[3 + (ord(char) % 5)] += 1.0
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            vectors.append(vector)

        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _as_list(value: TextInput) -> List[str]:
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _as_2d_float32(embedding: Any) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.ndim != 2:
            raise ValueError("embedding_method 必须返回一维或二维向量")
        return np.ascontiguousarray(embedding)

    def _route_names_key(self) -> str:
        return f"{self.name}:route_names"

    def _examples_key(self) -> str:
        return f"{self.name}:examples"

    def _embedding_key(self, index: int) -> str:
        return f"{self.name}:emb:{index}"

    def _pipe_set(self, pipe, key: str, value):
        """根据 ttl 决定使用 set 还是 setex。路由配置通常不需要过期。"""
        if self.ttl is None:
            pipe.set(key, value)
        else:
            pipe.setex(key, self.ttl, value)

    def _save_metadata(self):
        """保存 route 名称和示例问题列表。"""
        with self.redis.pipeline() as pipe:
            self._pipe_set(
                pipe,
                self._route_names_key(),
                json.dumps(self._route_names, ensure_ascii=False),
            )
            self._pipe_set(
                pipe,
                self._examples_key(),
                json.dumps(self._examples, ensure_ascii=False),
            )
            pipe.execute()

    def _load_from_redis(self):
        """从 Redis 恢复 route 名称、示例问题和 embedding，并重建 FAISS。"""
        try:
            raw_names = self.redis.get(self._route_names_key())
            if not raw_names:
                return

            raw_examples = self.redis.get(self._examples_key())
            saved_names = json.loads(raw_names)
            saved_examples = json.loads(raw_examples) if raw_examples else [""] * len(saved_names)

            embeddings = []
            route_names = []
            examples = []

            for index, route_name in enumerate(saved_names):
                cached = self.redis.get(self._embedding_key(index))
                if cached is None:
                    continue
                embeddings.append(np.frombuffer(cached, dtype=np.float32).copy())
                route_names.append(route_name)
                examples.append(saved_examples[index] if index < len(saved_examples) else "")

            if not embeddings:
                return

            embedding_matrix = np.array(embeddings, dtype=np.float32)
            self._dim = embedding_matrix.shape[1]
            self.index = faiss.IndexFlatL2(self._dim)
            self.index.add(embedding_matrix)
            self._route_names = route_names
            self._examples = examples
        except Exception as e:
            print(f"Load router state error: {e}")

    def add_route(self, questions: List[str], target: str):
        """
        注册路由。

        参数：
            questions: 该路由的示例问题列表。
            target: 路由名称，例如 greeting、refund、logistics。
        """
        questions = self._as_list(questions)
        if not questions:
            return []

        embeddings = self._as_2d_float32(self.embedding_method(questions))
        if embeddings.shape[0] != len(questions):
            raise ValueError("embedding 数量必须和 questions 数量一致")

        if self.index is None:
            self._dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self._dim)

        if self.index.d != embeddings.shape[1]:
            raise ValueError(f"embedding 维度应为 {self.index.d}，实际为 {embeddings.shape[1]}")

        start_index = len(self._route_names)
        self.index.add(embeddings)
        self._route_names.extend([target] * len(questions))
        self._examples.extend(questions)

        try:
            with self.redis.pipeline() as pipe:
                for offset, embedding in enumerate(embeddings):
                    self._pipe_set(
                        pipe,
                        self._embedding_key(start_index + offset),
                        embedding.astype(np.float32).tobytes(),
                    )
                self._pipe_set(
                    pipe,
                    self._route_names_key(),
                    json.dumps(self._route_names, ensure_ascii=False),
                )
                self._pipe_set(
                    pipe,
                    self._examples_key(),
                    json.dumps(self._examples, ensure_ascii=False),
                )
                return pipe.execute()
        except Exception as e:
            print(f"Add route error: {e}")
            return -1

    def route_with_score(self, question: str):
        """
        返回最匹配路由及距离分数。

        返回：
            (route_name, distance)
            distance 越小越相似；没有可用路由时返回 (None, inf)。
        """
        if self.index is None or not self._route_names:
            return (None, float("inf"))

        embedding = self._as_2d_float32(self.embedding_method(question))
        distances, indices = self.index.search(embedding, k=1)

        matched_index = int(indices[0][0])
        distance = float(distances[0][0])

        if matched_index < 0 or matched_index >= len(self._route_names):
            return (None, float("inf"))

        if self.distance_threshold is not None and distance > self.distance_threshold:
            return (None, distance)

        return (self._route_names[matched_index], distance)

    def route(self, question: str) -> Optional[str]:
        """返回最匹配的 route 名称。"""
        route_name, _ = self.route_with_score(question)
        return route_name

    def __call__(self, question: str) -> Optional[str]:
        """允许直接 router(question)，兼容原始作业示例。"""
        return self.route(question)

    def get_routes(self) -> List[str]:
        """返回去重后的 route 名称，并保留首次添加时的顺序。"""
        routes = []
        for route_name in self._route_names:
            if route_name not in routes:
                routes.append(route_name)
        return routes

    def remove_route(self, target: str):
        """
        删除指定 route，并重建 FAISS 索引。

        FAISS IndexFlatL2 不支持原地删除某条向量，所以删除 route 后需要用
        保留下来的向量重新创建索引。
        """
        if self.index is None:
            return 0

        keep_indices = [
            index
            for index, route_name in enumerate(self._route_names)
            if route_name != target
        ]

        removed_count = len(self._route_names) - len(keep_indices)
        if removed_count == 0:
            return 0

        kept_embeddings = [self.index.reconstruct(int(index)) for index in keep_indices]
        kept_route_names = [self._route_names[index] for index in keep_indices]
        kept_examples = [self._examples[index] for index in keep_indices]
        old_count = len(self._route_names)

        self._route_names = kept_route_names
        self._examples = kept_examples

        if kept_embeddings:
            embedding_matrix = np.array(kept_embeddings, dtype=np.float32)
            self._dim = embedding_matrix.shape[1]
            self.index = faiss.IndexFlatL2(self._dim)
            self.index.add(embedding_matrix)
        else:
            self.index = None
            self._dim = None

        with self.redis.pipeline() as pipe:
            for index in range(old_count):
                pipe.delete(self._embedding_key(index))
            for index, embedding in enumerate(kept_embeddings):
                self._pipe_set(
                    pipe,
                    self._embedding_key(index),
                    np.asarray(embedding, dtype=np.float32).tobytes(),
                )
            self._pipe_set(
                pipe,
                self._route_names_key(),
                json.dumps(self._route_names, ensure_ascii=False),
            )
            self._pipe_set(
                pipe,
                self._examples_key(),
                json.dumps(self._examples, ensure_ascii=False),
            )
            pipe.execute()

        return removed_count

    def clear_routes(self):
        """清空所有路由，方便重复运行测试。"""
        count = len(self._route_names)

        with self.redis.pipeline() as pipe:
            for index in range(count):
                pipe.delete(self._embedding_key(index))
            pipe.delete(self._route_names_key())
            pipe.delete(self._examples_key())
            pipe.execute()

        self.index = None
        self._dim = None
        self._route_names = []
        self._examples = []


if __name__ == "__main__":
    router = SemanticRouter(
        name="test_router",
        redis_url="localhost",
        distance_threshold=0.01,
    )

    router.clear_routes()

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "你好"],
        target="greeting",
    )
    router.add_route(
        questions=["如何退货", "我要退款"],
        target="refund",
    )
    router.add_route(
        questions=["帮我查物流", "我的快递到哪了"],
        target="logistics",
    )

    print("=== 路由测试 ===")
    print(f"'Hi, good morning' -> {router.route('Hi, good morning')}")
    print(f"'如何退货？'        -> {router.route('如何退货？')}")
    print(f"'帮我查物流'        -> {router.route('帮我查物流')}")
    print(f"'你好呀'            -> {router.route('你好呀')}")
    print(f"'随便聊聊'          -> {router.route('随便聊聊')}")

    print("\n=== 路由测试（带分数） ===")
    print(f"'Hi, good morning' -> {router.route_with_score('Hi, good morning')}")
    print(f"'我要退款'          -> {router.route_with_score('我要退款')}")

    print(f"\n所有路由: {router.get_routes()}")

    print("\n=== 删除 refund 后 ===")
    print(f"删除数量: {router.remove_route('refund')}")
    print(f"所有路由: {router.get_routes()}")
