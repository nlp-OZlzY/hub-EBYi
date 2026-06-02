"""AI 代理基类

所有 Agent（玩家代理、总结代理等）的基类，封装 LLM 客户端的初始化和基本对话能力。
"""

from llm.client import LLMClient


class BaseAgent:
    """AI 代理基类

    提供 LLM 客户端初始化和通用对话接口，子类可覆盖 system_prompt 定制行为。
    """

    def __init__(self):
        """初始化代理，创建 LLM 客户端实例"""
        self.client = LLMClient.from_config()
        self.system_prompt = "你好"

    async def run(self, input: str) -> str:
        """执行一次对话

        Args:
            input: 用户输入消息

        Returns:
            LLM 生成的回复文本
        """
        result = await self.client.chat(system=self.system_prompt, user=input)
        return result
