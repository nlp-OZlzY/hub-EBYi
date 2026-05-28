"""猎人 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class HunterAgent(BaseAgent):
    """猎人 Agent

    策略：
    1. 寻找狼人
    2. 死亡时带走狼人
    3. 适度暴露身份威慑
    4. 使用信息处理器辅助决策
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.can_shoot = True
        self.suspected_wolves: list[str] = []

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行猎人行动"""
        if game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)
        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def decide_shoot(self, game_state: GameState) -> Optional[str]:
        """决定是否开枪及带走谁"""
        if not self.can_shoot:
            return None

        # 优先带走狼人
        if self.suspected_wolves:
            for wolf_id in self.suspected_wolves:
                if game_state.players[wolf_id].alive:
                    return wolf_id

        return None

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言"""
        info_processor = context.get("info_processor")

        # 使用信息处理器分析可能的狼人
        if info_processor:
            suspicious = info_processor.get_most_suspicious(exclude=[self.agent_id])
            if suspicious:
                if suspicious not in self.suspected_wolves:
                    self.suspected_wolves.append(suspicious)

        # 生成发言
        speech_parts = []

        if self.suspected_wolves and game_state.day >= 2:
            suspect = game_state.players[self.suspected_wolves[0]]
            speech_parts.append(f"我观察 {suspect.name} 的发言风格，感觉比较可疑。")

        speech_parts.append("大家畅所欲言，共同分析。")

        speech = " ".join(speech_parts)

        return {
            "action": "speech",
            "content": speech,
            "suspected_wolves": self.suspected_wolves.copy()
        }

    def _act_vote(self, game_state: GameState, context: dict) -> dict:
        """投票"""
        info_processor = context.get("info_processor")

        # 优先投已识别的可疑狼人
        for wolf_id in self.suspected_wolves:
            if game_state.players[wolf_id].alive:
                return {
                    "action": "vote",
                    "target": wolf_id,
                    "target_name": game_state.players[wolf_id].name
                }

        # 使用信息处理器
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