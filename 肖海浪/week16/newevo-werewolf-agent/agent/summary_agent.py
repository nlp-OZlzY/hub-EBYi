"""总结代理模块

在游戏结束后，为每个玩家生成结构化表现总结（JSON 格式），
包含整体表现、策略评估、失误分析和改进建议。
总结结果保存到经验系统，供后续游戏的 PlayerAgent 参考。
"""

import json
import re
from typing import Dict, Optional

from llm.client import LLMClient


SUMMARY_PROMPT_TEMPLATE = """你刚刚完成了一局狼人杀游戏。请以第一人称视角，回顾你这局游戏的表现，进行深度总结。

## 你的身份
- 角色：{role_name}
- 阵营：{camp_name}
- 玩家名称：{player_name}

## 游戏结果
- 胜利方：{winner_name}
- 你的阵营：{is_winner_text}

## 你的游戏经历
以下是你在整局游戏中经历的所有事件：
{personal_history}

## 要求
请反思你的表现，并以 **JSON 格式** 输出以下内容（四个字段值都必须是自然语言句子，不要嵌套 JSON）：

```json
{{
    "summary": "整体表现总结（2-3句话）",
    "strategies": "使用了哪些策略，哪些有效、哪些无效",
    "mistakes": "犯了哪些错误，可以如何改进",
    "lessons": "对未来自己的 1-2 条具体建议"
}}
```

只输出一个 JSON 对象，不要 Markdown 说明，不要代码块包裹。
"""

# 总结 JSON 需完整四个字段，略提高上限避免截断导致解析失败
SUMMARY_MAX_COMPLETION_TOKENS = 512
MAX_PERSONAL_HISTORY_CHARS = 4000

SUMMARY_FIELDS = ("summary", "strategies", "mistakes", "lessons")


class SummaryAgent:
    """总结代理

    游戏结束后为每个玩家生成结构化复盘总结（JSON 格式），
    包含四个维度：整体表现、策略评估、失误分析、改进建议。

    总结以第一人称视角生成（"你"="该玩家"），并保存到经验系统。
    后续对局中，PlayerAgent 会将最近 3 条经验注入 prompt，实现经验积累。

    JSON 解析同样采用四层容错策略（与 PlayerAgent 一致），
    确保 LLM 输出格式不完美时仍能提取有效信息。
    """

    def __init__(self):
        self.client = LLMClient.from_config()
        self.system_prompt = (
            "你是一个狼人杀游戏的复盘分析师。"
            "你只输出一个合法的 JSON 对象，包含 summary、strategies、mistakes、lessons 四个字符串字段。"
        )

    async def generate_summary(
        self,
        player_name: str,
        role_name: str,
        camp: str,
        winner: Optional[str],
        personal_history: str,
    ) -> Dict[str, str]:
        """为一个玩家生成总结"""
        camp_name = "善良阵营" if camp == "good" else "邪恶阵营"
        winner_name = (
            "善良阵营（好人）"
            if winner == "good"
            else "邪恶阵营（狼人）"
            if winner == "evil"
            else (winner or "未知")
        )
        is_winner_text = "胜利！" if camp == winner else "失败。"

        if len(personal_history) > MAX_PERSONAL_HISTORY_CHARS:
            personal_history = (
                personal_history[-MAX_PERSONAL_HISTORY_CHARS:]
                + "\n...(较早经历已省略)"
            )

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            role_name=role_name,
            camp_name=camp_name,
            player_name=player_name,
            winner_name=winner_name,
            is_winner_text=is_winner_text,
            personal_history=personal_history,
        )

        try:
            result = await self.client.chat(
                system=self.system_prompt,
                user=prompt,
                temperature=0.5,
                max_completion_tokens=SUMMARY_MAX_COMPLETION_TOKENS,
            )
            return self._normalize_fields(self._parse_json_output(result))
        except Exception as e:
            return self._empty_result(f"总结生成失败: {e}")

    @staticmethod
    def _empty_result(summary: str = "") -> Dict[str, str]:
        return {
            "summary": summary,
            "strategies": "",
            "mistakes": "",
            "lessons": "",
        }

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _find_balanced_json(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @classmethod
    def _extract_fields_by_regex(cls, text: str) -> Dict[str, str]:
        """从残缺 JSON 中按字段提取字符串值"""
        result: Dict[str, str] = {}
        for field in SUMMARY_FIELDS:
            pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    result[field] = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    result[field] = match.group(1)
        return result

    @classmethod
    def _normalize_fields(cls, data: Dict[str, str]) -> Dict[str, str]:
        """确保四字段为纯文本；若 summary 里误存了整段 JSON 则拆开"""
        normalized = cls._empty_result()

        for key in SUMMARY_FIELDS:
            val = data.get(key, "")
            if val is None:
                val = ""
            normalized[key] = str(val).strip()

        summary = normalized["summary"]
        if summary.startswith("{") and '"summary"' in summary:
            reparsed = cls._parse_json_output(summary)
            reparsed = cls._normalize_fields(reparsed)
            for key in SUMMARY_FIELDS:
                if reparsed.get(key) and not normalized.get(key):
                    normalized[key] = reparsed[key]
                elif reparsed.get(key) and normalized.get(key, "").startswith("{"):
                    normalized[key] = reparsed[key]

        return normalized

    def _parse_json_output(self, output: str) -> Dict[str, str]:
        """解析 LLM 输出的 JSON（支持截断、代码块、嵌套引号）"""
        if not output or not output.strip():
            return self._empty_result()

        cleaned = self._strip_markdown_fence(output)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        json_str = self._find_balanced_json(cleaned)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        partial = self._extract_fields_by_regex(cleaned)
        if partial.get("summary"):
            for key in SUMMARY_FIELDS:
                partial.setdefault(key, "")
            return partial

        # 完全无法解析：不要把 JSON 键名原样展示，只保留可读片段
        plain = cleaned
        if plain.startswith("{"):
            m = re.search(r'"summary"\s*:\s*"([^"]+)', plain)
            if m:
                return self._empty_result(m.group(1))
            return self._empty_result("本局复盘生成不完整，请查看后台日志。")

        return self._empty_result(plain[:400])


__all__ = ["SummaryAgent"]
