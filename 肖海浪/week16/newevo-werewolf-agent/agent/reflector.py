"""自反思器模块

自演化系统的核心组件：在每局游戏结束后，根据角色表现指标和游戏日志，
由 LLM 评估当前 prompt 效果并生成改进版本。

工作流程：
1. 接收角色的当前 prompt、量化指标、游戏日志
2. 调用 LLM（使用反思器元提示词）分析优劣
3. 输出完整的改进后 prompt
"""

import os
from typing import Dict, Any

from llm.client import LLMClient


# 内置的反思器元提示词（当 prompts/meta/reflector.md 不存在时使用）
REFLECTOR_META_PROMPT = """你是一个 Agent 自演化系统的反思器。
你的任务是：评估当前角色 prompt 的效果，并生成改进版本。

你将收到：
1. 当前 prompt（正在使用的策略指令）
2. 本局游戏表现数据（胜负、关键决策、失误点）
3. 该角色视角的游戏日志

你需要：
1. 分析当前 prompt 中哪些指令导致了好的决策
2. 分析哪些指令导致了失误或次优决策
3. 针对性地修改 prompt，保留有效部分，改进无效部分
4. 输出完整的改进后 prompt（不要输出 diff，输出完整文件）

关键约束：
- 小幅度修改，保留有效策略，只调整有问题的部分
- 保持 prompt 的结构（标题、段落格式不变）
- 不要删除游戏规则相关内容（如"不能自救"、"单夜不能双药"等）
- 如果本局胜利，尽量保留现有策略，只做微调
- 如果本局失败，重点分析失败原因并针对性改进

输出格式：
直接输出完整的改进后 prompt 文件内容（Markdown 格式），不要包含其他解释文字。
"""


def _load_reflector_meta_prompt() -> str:
    """加载反思器元提示词，优先从文件加载"""
    meta_path = os.path.join("prompts", "meta", "reflector.md")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return f.read()
    return REFLECTOR_META_PROMPT


class SelfReflector:
    """自反思器

    自演化系统的核心：每局游戏结束后，将角色的当前 prompt、量化指标
    和游戏日志发给 LLM，让 LLM 分析哪些策略有效、哪些需要改进，
    最终输出一份改进后的完整 prompt。

    调用链：GameEngine → MetricsCollector → SelfReflector → PromptStore
    """

    def __init__(self):
        self.client = LLMClient.from_config()
        self.meta_prompt = _load_reflector_meta_prompt()

    async def reflect(
        self,
        role: str,
        current_prompt: str,
        metrics: Dict[str, Any],
        game_logs: str,
    ) -> str:
        """执行自反思，生成改进后的 prompt"""
        user_message = f"""## 当前 Prompt（角色：{role}）

{current_prompt}

## 本局游戏表现数据

{self._format_metrics(metrics)}

## 游戏日志

{game_logs}

请根据以上信息，输出改进后的完整 prompt。"""

        result = await self.client.chat(
            system=self.meta_prompt,
            user=user_message,
            temperature=0.7,
        )

        return result

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """格式化指标为可读文本"""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, bool):
                lines.append(f"- {key}: {'是' if value else '否'}")
            elif isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)


__all__ = ["SelfReflector"]
