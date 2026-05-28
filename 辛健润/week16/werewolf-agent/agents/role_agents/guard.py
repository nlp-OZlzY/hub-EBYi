"""守卫 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class GuardAgent(BaseAgent):
    """守卫 Agent

    策略：
    1. 优先守护预言家
    2. 守护自己
    3. 避免连续守同一人
    4. 使用信息处理器辅助决策
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.last_guard_target: Optional[str] = None
        self.guard_history: list[str] = []

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行守卫行动"""
        if game_state.phase.value == "night":
            return self._act_night(game_state, context)

        elif game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)

        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def _act_night(self, game_state: GameState, context: dict) -> dict:
        """夜间行动 - 守人"""
        alive = [p for p in game_state.players.values() if p.alive and p.id != self.agent_id]

        # 选择守护目标
        target = self._choose_guard_target(alive, game_state, context)

        if target:
            self.last_guard_target = target
            self.guard_history.append(target)
            return {
                "action": "guard",
                "target": target,
                "target_name": game_state.players[target].name
            }

        return {"action": "skip"}

    def _choose_guard_target(
        self,
        candidates: list,
        game_state: GameState,
        context: dict
    ) -> Optional[str]:
        """选择守护目标"""
        # 优先守护预言家
        seers = [p for p in candidates if p.role.role_type == RoleType.SEER]
        if seers and self.last_guard_target != seers[0].id:
            return seers[0].id

        # 其次守护女巫
        witches = [p for p in candidates if p.role.role_type == RoleType.WITCH]
        if witches and self.last_guard_target != witches[0].id:
            return witches[0].id

        # 守护自己
        if self.last_guard_target != self.agent_id:
            return self.agent_id

        # 使用信息处理器找到最可疑但不是狼人的玩家，守护他们
        info_processor = context.get("info_processor")
        if info_processor:
            # 守护被投票最多的玩家（可能是神职需要保护）
            alive_ids = [p.id for p in candidates]
            most_suspicious = info_processor.get_most_suspicious(exclude=[self.agent_id])
            if most_suspicious and most_suspicious in alive_ids and most_suspicious != self.last_guard_target:
                return most_suspicious

        # 守护其他好人（避免连续守同一人）
        for player in candidates:
            if player.id != self.last_guard_target and player.role.camp == Camp.GOOD:
                return player.id

        return None

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言"""
        info_processor = context.get("info_processor")

        speech_parts = []

        # 基于信息处理器添加可疑玩家
        if info_processor:
            suspicious = info_processor.get_most_suspicious(exclude=[self.agent_id])
            if suspicious:
                target_name = game_state.players[suspicious].name
                speech_parts.append(f"我注意到 {target_name} 的发言有些可疑。")

        if not speech_parts:
            speech_parts.append("今天大家好好发言，找出狼人。")

        speech = " ".join(speech_parts)

        return {
            "action": "speech",
            "content": speech
        }

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