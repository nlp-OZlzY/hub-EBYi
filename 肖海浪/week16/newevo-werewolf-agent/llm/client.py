"""LLM 客户端模块

封装 OpenAI 兼容 API（小米 MiMo 大模型），提供统一的异步对话接口。
所有需要调用 LLM 的模块（PlayerAgent、SelfReflector、SummaryAgent）都通过此客户端交互。

限流策略（双保险）：
1. 全局信号量限制并发数为 3，防止同时发起过多请求
2. 最小请求间隔 0.5 秒，平滑请求节奏
3. 遇到 429 限流时指数退避重试（2→4→8→16→32 秒）
"""

import asyncio
import json
import os
import time
from openai import AsyncOpenAI

# 全局信号量：限制同时并发的 LLM 请求数，避免触发 API 限流
_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(3)

# 全局最小请求间隔（秒），防止短时间内发送过多请求
_last_request_time = 0.0
_MIN_INTERVAL = 0.5  # 每次请求间隔至少 0.5 秒（总结阶段并行时缩短等待）


def load_llm_config(config_path: str = "config/llm_config.json") -> dict:
    """加载 LLM 配置文件

    Args:
        config_path: 配置文件路径，默认 config/llm_config.json

    Returns:
        包含 base_url、model、max_completion_tokens 等字段的配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class LLMClient:
    """LLM 异步客户端

    基于 OpenAI SDK 封装，支持小米 MiMo 等兼容 API。
    提供 from_config 工厂方法从配置文件创建实例，以及 chat 方法进行异步对话。
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_completion_tokens: int = 1024, temperature: float = 1.0, top_p: float = 0.95):
        """初始化 LLM 客户端

        Args:
            api_key: API 密钥
            base_url: API 基础地址
            model: 模型名称
            max_completion_tokens: 最大生成 token 数
            temperature: 采样温度，越高越随机
            top_p: 核采样参数
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p

    @classmethod
    def from_config(cls, config_path: str = "config/llm_config.json", api_key: str = None) -> "LLMClient":
        """从配置文件创建 LLMClient 实例

        优先级：传入参数 > 环境变量 MIMO_API_KEY > 配置文件 api_key 字段

        Args:
            config_path: 配置文件路径
            api_key: API 密钥（可选）

        Returns:
            LLMClient 实例
        """
        config = load_llm_config(config_path)
        if api_key is None:
            api_key = os.environ.get("MIMO_API_KEY") or config.get("api_key", "")
        return cls(
            api_key=api_key,
            base_url=config["base_url"],
            model=config["model"],
            max_completion_tokens=config.get("max_completion_tokens", 1024),
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 0.95),
        )

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = None,
        max_completion_tokens: int = None,
    ) -> str:
        """发送对话请求并获取回复（带限流和重试）

        内置并发控制（信号量）和请求间隔保护，遇到 429 限流时自动重试。

        Args:
            system: 系统提示词（角色设定）
            user: 用户消息内容
            temperature: 临时覆盖的采样温度（可选）
            max_completion_tokens: 临时覆盖的最大生成 token 数（可选）

        Returns:
            LLM 生成的回复文本
        """
        global _last_request_time

        max_retries = 5
        for attempt in range(max_retries):
            async with _CONCURRENCY_SEMAPHORE:
                # 控制请求间隔，避免短时间大量请求
                now = time.monotonic()
                wait = _MIN_INTERVAL - (now - _last_request_time)
                if wait > 0:
                    await asyncio.sleep(wait)
                _last_request_time = time.monotonic()

                try:
                    tokens = (
                        max_completion_tokens
                        if max_completion_tokens is not None
                        else self.max_completion_tokens
                    )
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_completion_tokens=tokens,
                        temperature=temperature if temperature is not None else self.temperature,
                        top_p=self.top_p,
                    )
                    content = response.choices[0].message.content
                    return content if content is not None else ""
                except Exception as e:
                    error_str = str(e)
                    # 429 限流或队列满：等待后重试
                    if "429" in error_str or "rate limit" in error_str.lower() or "queue" in error_str.lower():
                        wait_time = 2 ** attempt * 2  # 2, 4, 8, 16, 32 秒
                        print(f"[LLM] 限流，第 {attempt+1} 次重试，等待 {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    # 其他错误直接抛出
                    raise

        # 所有重试都失败
        raise RuntimeError("LLM 请求失败：超过最大重试次数（API 持续限流）")
