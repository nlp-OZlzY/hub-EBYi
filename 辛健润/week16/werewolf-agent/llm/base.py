"""LLM 抽象接口层"""
from abc import ABC, abstractmethod
from typing import Optional
import os


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pass

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """对话模式"""
        pass


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 实现"""

    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def _call_api(self, messages: list[dict], **kwargs) -> str:
        """调用 DeepSeek API"""
        import urllib.request
        import urllib.error
        import json

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise Exception(f"DeepSeek API error: {e.code} - {error_body}")
        except Exception as e:
            raise Exception(f"DeepSeek API call failed: {str(e)}")

    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复（单轮对话）"""
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, **kwargs)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """对话模式（多轮）"""
        return self._call_api(messages, **kwargs)