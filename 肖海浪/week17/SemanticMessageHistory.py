"""
SemanticMessageHistory.py

功能：
    管理一轮会话中的历史消息，支持：
      - 追加消息
      - 获取完整历史
      - 按角色获取最近消息
      - 按文本相关性获取最相关消息
      - 只保留最近 N 条或清空历史

说明：
    redis-vl-python 中的 MessageHistory 更偏工程化，会结合 Redis 数据结构、
    过滤条件和向量检索。本作业用 Redis 保存 JSON 数组，再用
    Levenshtein.ratio 做轻量相关性排序，重点是理解“历史可检索”的设计。

Redis 数据结构：
    Key:   semantic_history:{session_id}
    Value: JSON 字符串，内容是消息列表
"""

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Union

import redis

try:
    import Levenshtein
except ImportError:
    Levenshtein = None


Message = Dict[str, Any]
RoleInput = Optional[Union[str, List[str]]]


class SemanticMessageHistory:
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
        self.key = f"semantic_history:{self.name}"
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """
        计算两个文本的相似度。

        优先使用 python-Levenshtein；如果环境里没装，则退回 Python 标准库
        difflib，保证文件至少能被导入和阅读。
        """
        if Levenshtein is not None:
            return Levenshtein.ratio(left, right)
        return SequenceMatcher(None, left, right).ratio()

    @staticmethod
    def _normalize_messages(message: Union[Message, List[Message]]) -> List[Message]:
        """单条 dict 用 append 语义，多条 list 用 extend 语义。"""
        if isinstance(message, dict):
            return [message]
        if isinstance(message, list):
            return message
        raise TypeError("message 必须是 dict 或 list[dict]")

    @staticmethod
    def _normalize_roles(role: RoleInput) -> Optional[set]:
        """role 支持字符串或字符串列表。"""
        if role is None:
            return None
        if isinstance(role, str):
            return {role}
        return set(role)

    def get_history(self) -> List[Message]:
        """获取完整历史；没有历史时返回空列表。"""
        history = self.redis.get(self.key)
        if not history:
            return []
        return json.loads(history)

    def add_message(self, message: Union[Message, List[Message]]):
        """
        追加消息。

        消息格式示例：
            {"role": "user", "content": "你好"}
            {"role": "llm", "content": "你好，有什么可以帮你？"}

        注意：
            原始半成品中直接 history.extend(message)。如果 message 是 dict，
            Python 会把 dict 的 key 当成可迭代对象写入，导致历史变成
            ["role", "content"]。这里先统一成 list[dict] 再 extend。
        """
        history = self.get_history()
        history.extend(self._normalize_messages(message))

        self.redis.setex(
            self.key,
            self.ttl,
            json.dumps(history, ensure_ascii=False),
        )

    def get_recent(self, role: RoleInput = None, top_k: Optional[int] = 10) -> List[Message]:
        """
        获取最近消息。

        参数：
            role: None 表示不过滤；字符串表示单角色；列表表示多个角色。
            top_k: 返回最近 N 条；None 表示返回全部。
        """
        history = self.get_history()
        roles = self._normalize_roles(role)

        if roles is not None:
            history = [message for message in history if message.get("role") in roles]

        if top_k is None:
            return history
        if top_k <= 0:
            return []
        return history[-top_k:]

    def get_relevant(self, content: str, top_k: Optional[int] = 10) -> List[Message]:
        """
        按文本相关性获取历史消息。

        Levenshtein.ratio 的取值范围是 0~1，越接近 1 表示越相似。
        这里不先做关键词包含过滤，因为相关文本未必包含完全相同的词。
        """
        history = self.get_history()
        if not history:
            return []

        scored_messages = []
        for message in history:
            message_content = str(message.get("content", ""))
            score = self._similarity(message_content, content)
            if score > 0:
                scored_messages.append((score, message))

        scored_messages.sort(key=lambda item: item[0], reverse=True)
        messages = [message for _, message in scored_messages]

        if top_k is None:
            return messages
        if top_k <= 0:
            return []
        return messages[:top_k]

    def delete_history(self, top_k: int = 10):
        """
        删除旧历史，只保留最近 top_k 条。

        top_k <= 0 时表示不保留任何历史。
        """
        history = self.get_history()
        history = history[-top_k:] if top_k > 0 else []

        self.redis.setex(
            self.key,
            self.ttl,
            json.dumps(history, ensure_ascii=False),
        )

    def clear_history(self):
        """清空整个会话历史。"""
        return self.redis.delete(self.key)


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )

    history.clear_history()

    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "今天北京天气怎么样？"},
    ])

    print("=== get_history ===")
    print(history.get_history())

    print("\n=== get_recent top_k=1 ===")
    print(history.get_recent(top_k=1))

    print("\n=== get_recent role=user ===")
    print(history.get_recent(role="user", top_k=2))

    print("\n=== get_recent role=['user', 'llm'] ===")
    print(history.get_recent(role=["user", "llm"], top_k=2))

    print("\n=== get_relevant 'weather today' ===")
    print(history.get_relevant("weather today", top_k=2))

    print("\n=== add_message 单条 dict ===")
    history.add_message({"role": "user", "content": "single message test"})
    print(history.get_recent(top_k=1))

    print("\n=== delete_history top_k=3 ===")
    history.delete_history(top_k=3)
    print(history.get_history())
