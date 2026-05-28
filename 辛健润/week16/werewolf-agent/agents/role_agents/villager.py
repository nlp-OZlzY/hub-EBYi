"""村民 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class VillagerAgent(BaseAgent):
    """村民 Agent

    策略：
    1. 通过发言逻辑判断狼人
    2. 配合好人阵营
    3. 利用 InformationProcessor 分析投票
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.suspicion_history: dict[str, int] = {}

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行村民行动"""
        if game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)
        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言 - 分析局势"""
        speeches = context.get("public_info", {}).get("public_speeches", [])
        alive = [p for p in game_state.players.values() if p.alive and p.id != self.agent_id]

        # 分析发言，更新怀疑列表
        self._analyze_speeches(speeches, game_state)

        # 获取信息处理器
        info_processor = context.get("info_processor")

        # 生成发言
        target = self._get_most_suspicious(info_processor)
        target_name = game_state.players[target].name if target and target in game_state.players else None

        speech_parts = []
        if target_name:
            speech_parts.append(f"我注意到 {target_name} 的发言有些可疑。")

        if game_state.day == 1:
            speech_parts.append("第一轮，大家先各抒己见。")
        else:
            speech_parts.append("今天我们需要找出狼人。")

        speech = " ".join(speech_parts)

        return {
            "action": "speech",
            "content": speech,
            "suspected": target
        }

    def _analyze_speeches(self, speeches: list, game_state: GameState):
        """分析发言，更新怀疑分数"""
        for speech in speeches:
            sender = speech.get("sender")
            content = speech.get("content", "")

            # 简单的关键词分析
            suspicious_keywords = ["紧张", "迟疑", "躲闪", "跟风", "狼"]
            for keyword in suspicious_keywords:
                if keyword in content and sender and sender != self.agent_id:
                    self.suspicion_history[sender] = self.suspicion_history.get(sender, 0) + 1

    def _get_most_suspicious(self, info_processor=None) -> Optional[str]:
        """获取最可疑玩家"""
        # 优先使用自己的分析
        if self.suspicion_history:
            return max(self.suspicion_history.items(), key=lambda x: x[1])[0]

        # 使用信息处理器的推荐
        if info_processor:
            return info_processor.get_most_suspicious(exclude=[self.agent_id])

        return None

    def _act_vote(self, game_state: GameState, context: dict) -> dict:
        """投票 - 使用信息处理器推荐"""
        info_processor = context.get("info_processor")

        # 使用信息处理器获取投票目标
        target = self._get_most_suspicious(info_processor)

        # 如果没有可疑目标，随机选择一个
        if not target:
            alive = [p.id for p in game_state.players.values()
                    if p.alive and p.id != self.agent_id]
            target = alive[0] if alive else None

        target_name = game_state.players[target].name if target and target in game_state.players else None

        return {
            "action": "vote",
            "target": target,
            "target_name": target_name
        }