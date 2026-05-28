"""女巫 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class WitchAgent(BaseAgent):
    """女巫 Agent

    策略：
    1. 第一晚通常救人
    2. 合理使用毒药
    3. 关键时刻救预言家
    4. 隐藏身份，适时使用毒药
    5. 使用信息处理器决策
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.has_heal = True
        self.has_poison = True
        self.healed_players: list[str] = []
        self.poisoned_players: list[str] = []
        self.first_night_healed = False
        self.suspicious_players: list[str] = []

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行女巫行动"""
        if game_state.phase.value == "night":
            return self._act_night(game_state, context)

        elif game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)

        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def _act_night(self, game_state: GameState, context: dict) -> dict:
        """夜间行动 - 用药"""
        actions = []

        # 检测是否有人需要救
        kill_target = self._get_kill_target(context)

        # 判断是否救人
        if kill_target and self.has_heal and not self.first_night_healed:
            # 第一晚通常救人
            actions.append({
                "action": "heal",
                "target": kill_target,
                "reason": "第一晚救人保命"
            })
            self.has_heal = False
            self.first_night_healed = True
            self.healed_players.append(kill_target)

        # 判断是否用毒（基于信息处理器）
        if self.has_poison and self.suspicious_players:
            target = self._choose_poison_target(game_state, context)
            if target:
                actions.append({
                    "action": "poison",
                    "target": target,
                    "reason": f"毒死 {game_state.players[target].name}"
                })
                self.has_poison = False
                self.poisoned_players.append(target)

        if not actions:
            return {"action": "skip"}

        return actions[0] if len(actions) == 1 else {"action": "multi", "sub_actions": actions}

    def _get_kill_target(self, context: dict) -> Optional[str]:
        """获取狼人刀人目标"""
        return context.get("night_kill_target")

    def _choose_poison_target(self, game_state: GameState, context: dict) -> Optional[str]:
        """选择毒人目标"""
        info_processor = context.get("info_processor")

        if info_processor:
            # 找到最可疑的非狼人
            alive = {p.id: p for p in game_state.players.values() if p.alive}
            target = info_processor.get_most_suspicious(exclude=[self.agent_id])
            if target and target in alive:
                return target

        # 默认选择可疑玩家列表中的
        for player_id in self.suspicious_players:
            if game_state.players[player_id].alive:
                return player_id

        return None

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言 - 伪装"""
        speeches = context.get("public_info", {}).get("public_speeches", [])

        # 分析发言，更新可疑玩家列表
        self._analyze_speeches(speeches, game_state)

        speech_prompt = f"""你是 {self.name}，在狼人杀游戏中。
你实际上是女巫，但需要伪装成好人发言。

当前局势：
- 存活玩家：{len([p for p in game_state.players.values() if p.alive])}
- 你的药水状态：{'已救人' if not self.has_heal else '有解药'}, {'已毒人' if not self.has_poison else '有毒药'}

请生成一段自然的发言（100字左右），不要暴露女巫身份。
"""

        speech = self.llm.chat([{"role": "user", "content": speech_prompt}])

        return {
            "action": "speech",
            "content": speech
        }

    def _analyze_speeches(self, speeches: list, game_state: GameState):
        """分析发言"""
        suspicious_keywords = ["紧张", "迟疑", "躲闪", "跟风", "可疑"]
        for speech in speeches:
            sender = speech.get("sender")
            content = speech.get("content", "")

            for kw in suspicious_keywords:
                if kw in content and sender and sender not in self.suspicious_players:
                    self.suspicious_players.append(sender)

    def _act_vote(self, game_state: GameState, context: dict) -> dict:
        """投票"""
        info_processor = context.get("info_processor")

        # 使用信息处理器获取投票目标
        if info_processor:
            target = info_processor.get_vote_target_recommendation(
                agent_id=self.agent_id,
                alive_players={p.id: p for p in game_state.players.values() if p.alive}
            )
            if target:
                return {
                    "action": "vote",
                    "target": target,
                    "target_name": game_state.players[target].name
                }

        # 默认投票
        alive = [p.id for p in game_state.players.values()
                if p.alive and p.id != self.agent_id]
        target = alive[0] if alive else None

        return {
            "action": "vote",
            "target": target,
            "target_name": game_state.players[target].name if target else None
        }