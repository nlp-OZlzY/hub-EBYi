"""玩家 AI 代理模块

每个玩家角色对应一个 AI 代理，负责：
- 从 prompt 文件加载角色策略（支持自演化更新）
- 根据游戏状态做出夜间行动、白天发言、投票等决策
- 支持多种决策风格（谨慎、大胆、随机、平衡）
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from llm.client import LLMClient
from schema.system_config import load_system_config
from memory.experience import get_experience_prompt

config = load_system_config("config/system_config.json")


# 决策风格定义
DECISION_STYLES = {
    "cautious": {
        "name": "谨慎型",
        "description": "宁可放过可疑玩家，也不轻易误杀好人",
        "speech_tendency": "保守分析，少说少错",
        "vote_tendency": "跟票，不首先提名",
        "night_tendency": "不轻易用药/查验",
    },
    "bold": {
        "name": "大胆型",
        "description": "敢于冒险，快速做出判断",
        "speech_tendency": "激进指控，主动带队",
        "vote_tendency": "果断投票，不怕误杀",
        "night_tendency": "冒险用药/深夜击杀",
    },
    "random": {
        "name": "随机型",
        "description": "不要过度分析，依靠直觉",
        "speech_tendency": "随机发言，看心情",
        "vote_tendency": "随机投票",
        "night_tendency": "随机目标",
    },
    "balanced": {
        "name": "平衡型",
        "description": "综合考虑各种因素",
        "speech_tendency": "客观分析",
        "vote_tendency": "理性分析后投票",
        "night_tendency": "合理使用能力",
    },
}


@dataclass
class _AgentDescriptor:
    """兼容旧代码和旧测试的轻量代理描述。"""

    name: str
    model: str
    instructions: str


class PlayerAgent:
    """玩家AI代理

    根据角色类型生成对应的LLM代理，进行游戏决策
    """

    def __init__(self, player_id: int, role_name: str, private_context: Dict[str, Any],
                 camp: str, decision_style: str = "balanced", role_type: str = "",
                 prompt_file: Optional[str] = None):
        self.player_id = player_id
        self.role_name = role_name
        self.private_context = private_context
        self.camp = camp
        self.decision_style = decision_style
        self.role_type = role_type or ""  # 英文角色类型，如 "werewolf", 用于加载经验
        self.prompt_file = prompt_file
        self.client = LLMClient.from_config()

        instructions = self._build_instructions()

        self.agent = _AgentDescriptor(
            name=f"Player_{player_id}",
            model=config.default_model,
            instructions=instructions,
        )

    def _build_instructions(self) -> str:
        """构建角色提示词（基于 Agent.md）"""
        agent_prompt = self._load_agent_prompt()
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])

        instructions = f"""{agent_prompt}

## 你的信息
- 玩家ID：{self.player_id}
- 角色：{self.role_name}
- 阵营：{"善良阵营" if self.camp == "good" else "邪恶阵营"}
- 私有信息：{self.private_context}

## 决策风格
{style_info['name']}：{style_info['description']}

{self._get_experience_section()}
"""
        return instructions

    def _load_agent_prompt(self) -> str:
        """加载角色 Agent.md 完整提示词

        优先级：prompt_file > prompts/agents/{role_type}_agent.md > prompts/roles/{role_type}.md
        """
        # 1. 指定的 prompt 文件
        if self.prompt_file and os.path.exists(self.prompt_file):
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read()

        if self.role_type:
            # 2. Agent.md（新格式）
            agent_path = os.path.join("prompts", "agents", f"{self.role_type}_agent.md")
            if os.path.exists(agent_path):
                with open(agent_path, "r", encoding="utf-8") as f:
                    return f.read()

            # 3. 旧格式 roles/*.md（兼容）
            from prompt_store.store import PromptStore
            content = PromptStore().read_prompt(self.role_type)
            if content:
                return content

        return "你是一个狼人杀游戏中的玩家。请根据当前游戏状态做出决策。"

    def _get_experience_section(self) -> str:
        """获取过往经验文本，用于注入到提示词中"""
        if not self.role_type:
            return ""
        return get_experience_prompt(self.role_type)

    async def decide_night_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """夜晚行动决策（含 target fallback）

        Args:
            game_state: 当前游戏状态

        Returns:
            决策结果，包含 action, target, reasoning
        """
        prompt = self._build_night_prompt(game_state)
        result = await self.client.chat(system=self._build_instructions(), user=prompt)
        decision = self._parse_json_output(result)

        # target 为 null 或缺失时，随机选一个合法目标
        if decision.get("target") is None:
            alive_others = [
                p["player_id"] for p in game_state.get("other_players", [])
                if p.get("is_alive", True) and p["player_id"] != self.player_id
            ]
            if alive_others:
                decision["target"] = random.choice(alive_others)

        return decision

    async def decide_speech(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """白天发言决策（含 content fallback）

        Args:
            game_state: 当前游戏状态

        Returns:
            决策结果，包含 action, content
        """
        prompt = self._build_speech_prompt(game_state)
        instructions = self._build_instructions()

        result = await self.client.chat(system=instructions, user=prompt)
        decision = self._parse_json_output(result)
        decision = self._ensure_speech_content(decision, result, game_state)

        # LLM 返回空/无效时重试一次
        if decision.get("_used_fallback"):
            result = await self.client.chat(system=instructions, user=prompt)
            retry = self._parse_json_output(result)
            retry = self._ensure_speech_content(retry, result, game_state)
            if not retry.get("_used_fallback"):
                return retry

        decision.pop("_used_fallback", None)
        return decision

    @staticmethod
    def _looks_like_json_fragment(text: str) -> bool:
        """判断文本是否为未解析完的 JSON 片段（不应直接展示为发言）"""
        t = str(text).strip()
        if not t:
            return False
        if t.startswith("{") or t.startswith("["):
            return True
        if '"action"' in t or '"content"' in t or '"target"' in t:
            return True
        return False

    def _ensure_speech_content(
        self,
        decision: Dict[str, Any],
        raw_result: str,
        game_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """补全发言 content；无法解析时标记 _used_fallback"""
        content = str(decision.get("content", "")).strip()
        if content and not self._looks_like_json_fragment(content):
            return decision
        if content:
            decision.pop("content", None)

        if raw_result and len(raw_result.strip()) > 2:
            clean = raw_result.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean).strip()
            clean = re.sub(r'^\{[^}]*"content"\s*:\s*"', "", clean)
            clean = re.sub(r'"\s*\}$', "", clean)
            if clean and len(clean.strip()) > 2 and not clean.strip().startswith("{"):
                decision["content"] = clean.strip()[:300]
                return decision
            if not clean.strip().startswith("{"):
                decision["content"] = raw_result.strip()[:300]
                return decision

        day = game_state.get("day", game_state.get("day_number", "?"))
        alive = [
            p["player_id"]
            for p in game_state.get("other_players", [])
            if p.get("is_alive", True)
        ]
        alive.append(self.player_id)
        decision["content"] = (
            f"第{day}天，目前场上还有{len(alive)}人存活。"
            f"我（玩家{self.player_id}）会继续根据发言和投票情况分析，请大家理性讨论。"
        )
        decision["_used_fallback"] = True
        if not raw_result or not raw_result.strip():
            print(
                f"[警告] 玩家{self.player_id} 发言：LLM 返回为空，已使用备用发言"
            )
        else:
            print(
                f"[警告] 玩家{self.player_id} 发言：JSON 解析失败，已使用备用发言。"
                f" 原始输出: {raw_result[:80]}..."
            )
        return decision

    async def decide_vote(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """投票决策（含 target fallback）

        Args:
            game_state: 当前游戏状态

        Returns:
            决策结果，包含 action, target
        """
        prompt = self._build_vote_prompt(game_state)
        result = await self.client.chat(system=self._build_instructions(), user=prompt)
        decision = self._parse_json_output(result)

        # target 为 null 或缺失时，随机选一个存活玩家
        if decision.get("target") is None:
            alive_others = [
                p["player_id"] for p in game_state.get("other_players", [])
                if p.get("is_alive", True) and p["player_id"] != self.player_id
            ]
            if alive_others:
                decision["target"] = random.choice(alive_others)

        return decision

    @staticmethod
    def _truncate_game_state(game_state: Dict[str, Any], max_dialogues: int = 12) -> Dict[str, Any]:
        """截断游戏状态，只保留最近 N 条对话

        随着游戏进行，对话历史会越来越长，直接传入 LLM 会超出 token 限制。
        此方法只保留最近 max_dialogues 条对话，在信息量和 prompt 长度之间取得平衡。
        """
        truncated = dict(game_state)
        dialogues = truncated.get("dialogues", [])
        if len(dialogues) > max_dialogues:
            truncated["dialogues"] = dialogues[-max_dialogues:]
            truncated["_dialogues_truncated"] = True
            truncated["_total_dialogues"] = len(dialogues)
        return truncated

    def _build_night_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建夜间决策提示（截断历史）"""
        state = self._truncate_game_state(game_state)
        return f"""## 当前游戏状态
{state}

请决定今晚的行动，输出JSON：
{{"action": "night_action", "target": 玩家ID数字, "reasoning": "理由"}}"""

    def _build_speech_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建白天发言提示（截断历史）"""
        state = self._truncate_game_state(game_state)
        return f"""## 当前游戏状态
{state}

现在是公开辩论时间，请发表你的发言，输出JSON：
{{"action": "speech", "content": "你的发言内容"}}"""

    def _build_vote_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建投票提示（截断历史）"""
        state = self._truncate_game_state(game_state)
        return f"""## 当前游戏状态
{state}

现在进入投票环节，输出JSON：
{{"action": "vote", "target": 玩家ID数字}}"""

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _find_balanced_json(text: str) -> Optional[str]:
        """从文本中提取首个平衡的大括号 JSON 对象（支持 content 内含括号）"""
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

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """解析 LLM 输出的 JSON（四层容错）

        LLM 输出不一定合法 JSON，因此采用逐层降级策略：
        第 1 层：json.loads 直接解析（最理想情况）
        第 2 层：平衡大括号提取（处理 LLM 在 JSON 前后添加多余文本的情况）
        第 3 层：正则提取关键字段（处理 JSON 格式损坏但字段可辨认的情况）
        第 4 层：将纯文本作为 content 返回（LLM 完全未遵循 JSON 格式）

        Returns:
            解析后的决策字典，保证至少有 action 字段
        """
        if not output or not output.strip():
            return {"action": "unknown", "raw_output": ""}

        cleaned = self._strip_markdown_fence(output)

        # 第 1 层：直接解析
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 第 2 层：平衡括号提取 JSON
        json_str = self._find_balanced_json(cleaned)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 第 3 层：正则提取关键字段
        result = {}

        # 提取 target（数字）
        target_match = re.search(r'"target"\s*:\s*(\d+)', output)
        if target_match:
            result["target"] = int(target_match.group(1))
        else:
            # 也尝试匹配 "target": 数字 的格式（无引号）
            target_match2 = re.search(r'target["\s:]*(\d+)', output, re.IGNORECASE)
            if target_match2:
                result["target"] = int(target_match2.group(1))

        # 提取 action
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', output)
        if action_match:
            result["action"] = action_match.group(1)

        # 提取 content（发言内容）
        content_match = re.search(r'"content"\s*:\s*"([^"]*)"', output)
        if content_match:
            result["content"] = content_match.group(1)

        # 提取 reasoning
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', output)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)

        if result:
            result.setdefault("action", "unknown")
            return result

        # 第 4 层：纯文本回复（排除 JSON 片段）
        if not self._looks_like_json_fragment(cleaned):
            return {"action": "unknown", "raw_output": output, "content": cleaned[:300]}
        return {"action": "unknown", "raw_output": output}


class JudgeAgent:
    """主持人AI代理（裁判）

    负责判定游戏规则、计算结果、推进游戏流程
    """

    def __init__(self):
        instructions = """你是狼人杀游戏的主持人（裁判）。
你的职责是：
1. 按照游戏规则推进游戏流程
2. 收集并执行玩家的决策
3. 宣布每天的死亡结果
4. 判断游戏是否结束及胜利方

游戏流程：
1. 夜晚：依次执行狼人杀人、预言家查验、女巫用药
2. 白天：宣布死亡、公开辩论、投票处决

你必须输出JSON格式的游戏指令。
"""
        self.agent = _AgentDescriptor(
            name="Judge",
            model=config.default_model,
            instructions=instructions,
        )

    async def announce_death(self, deaths: List[int], cause: str, game_state: Dict[str, Any]) -> str:
        """宣布死亡结果

        Args:
            deaths: 死亡玩家ID列表
            cause: 死亡原因（night_kill, vote, shoot, poison）

        Returns:
            死亡公告文本
        """
        if not deaths:
            return "今晚无人死亡。"

        death_names = [f"玩家{p}" for p in deaths]
        cause_desc = {
            "night_kill": "昨夜",
            "vote": "投票",
            "shoot": "枪杀",
            "poison": "毒杀",
        }.get(cause, cause)

        announcement = f"{cause_desc}，以下玩家死亡：{', '.join(death_names)}"

        # 添加死亡玩家遗言
        for player_id in deaths:
            player = game_state.get("players", {}).get(player_id)
            if player:
                announcement += f"\n{player.get('name', f'玩家{player_id}')} 说："

        return announcement

    async def announce_phase(self, phase: str, day_number: int) -> str:
        """宣布游戏阶段

        Args:
            phase: 当前阶段
            day_number: 第几天

        Returns:
            阶段公告文本
        """
        if "night" in phase:
            return f"第{day_number}夜开始，请各位保持安静。"
        else:
            return f"第{day_number}天，阳光照耀，请各位玩家开始发言。"


def create_player_agent(player_id: int, role_name: str, private_context: Dict[str, Any],
                       camp: str, decision_style: str = "balanced",
                       role_type: str = "", prompt_file: Optional[str] = None) -> PlayerAgent:
    """工厂函数：创建玩家代理"""
    return PlayerAgent(
        player_id,
        role_name,
        private_context,
        camp,
        decision_style,
        role_type,
        prompt_file,
    )


def create_judge_agent() -> JudgeAgent:
    """工厂函数：创建主持人代理

    Returns:
        JudgeAgent实例
    """
    return JudgeAgent()
