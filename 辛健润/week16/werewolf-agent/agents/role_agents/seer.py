"""预言家 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class SeerAgent(BaseAgent):
    """预言家 Agent

    策略：
    1. 每晚查验可疑玩家
    2. 找准时机公开查验结果
    3. 保护自己，避免被刀
    4. 优先投票给已查验的狼人
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.verified_players: dict[str, bool] = {}  # player_id -> is_wolf
        self.public_reveal_round: Optional[int] = None

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行预言家行动"""
        if game_state.phase.value == "night":
            return self._act_night(game_state, context)
        elif game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)
        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def _act_night(self, game_state: GameState, context: dict) -> dict:
        """夜间行动 - 验人"""
        alive = [p for p in game_state.players.values() if p.alive]

        # 选择查验目标（优先查验未验证的）
        unverified = [
            p for p in alive
            if p.id != self.agent_id and p.id not in self.verified_players
        ]

        if not unverified:
            return {"action": "skip"}

        # 根据发言和投票选择可疑目标
        target = self._choose_verify_target(unverified, game_state, context)

        return {
            "action": "verify",
            "target": target,
            "target_name": game_state.players[target].name
        }

    def _choose_verify_target(
        self,
        candidates: list,
        game_state: GameState,
        context: dict
    ) -> str:
        """选择查验目标"""
        info_processor = context.get("info_processor")

        # 优先查验信息处理器推荐的可疑玩家
        if info_processor:
            recommended = info_processor.get_most_suspicious(exclude=[self.agent_id])
            if recommended:
                return recommended

        # 默认选择第一个候选
        return candidates[0].id

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言"""
        verified = self.verified_players

        # 决定是否公开查验结果
        should_reveal = self._should_reveal(game_state.day, len(verified))

        speech_parts = []

        if should_reveal:
            # 公开查验到的狼人
            wolves = [pid for pid, is_wolf in verified.items() if is_wolf]
            if wolves:
                target_name = game_state.players[wolves[0]].name
                speech_parts.append(f"昨晚我查验了 {target_name}，他是狼人！")

        if not speech_parts:
            # 分析发言找出可疑玩家
            suspicious = self._find_suspicious_from_speech(game_state, context)
            if suspicious:
                speech_parts.append(f"我观察 {game_state.players[suspicious].name} 的发言有些可疑。")
            else:
                speech_parts.append("今天大家好好发言，找出狼人。")

        speech = " ".join(speech_parts)

        return {
            "action": "speech",
            "content": speech,
            "revealed": should_reveal
        }

    def _find_suspicious_from_speech(self, game_state: GameState, context: dict) -> Optional[str]:
        """从发言中找可疑玩家"""
        speeches = context.get("public_info", {}).get("public_speeches", [])

        suspicious_keywords = ["可疑", "不像", "紧张", "躲闪"]
        scores = {}

        for speech in speeches:
            sender = speech.get("sender")
            content = speech.get("content", "")

            for kw in suspicious_keywords:
                if kw in content and sender and sender != self.agent_id:
                    scores[sender] = scores.get(sender, 0) + 1

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return None

    def _should_reveal(self, day: int, verified_count: int) -> bool:
        """判断是否应该公开"""
        # 第3天或查验到2个狼人后公开
        return day >= 3 or verified_count >= 2

    def _act_vote(self, game_state: GameState, context: dict) -> dict:
        """投票 - 优先投已查验的狼人"""
        # 投票给已知的狼人
        wolves = [
            pid for pid, is_wolf in self.verified_players.items()
            if is_wolf and game_state.players[pid].alive
        ]

        target = wolves[0] if wolves else None

        # 如果没有已知狼人，使用信息处理器
        if not target:
            info_processor = context.get("info_processor")
            if info_processor:
                target = info_processor.get_vote_target_recommendation(
                    agent_id=self.agent_id,
                    alive_players={p.id: p for p in game_state.players.values() if p.alive},
                    known_wolves=wolves
                )

        target_name = game_state.players[target].name if target and target in game_state.players else None

        return {
            "action": "vote",
            "target": target,
            "target_name": target_name
        }

    def record_verify_result(self, target_id: str, is_wolf: bool):
        """记录查验结果"""
        self.verified_players[target_id] = is_wolf

    def get_verified_info(self) -> dict:
        """获取查验信息"""
        return self.verified_players.copy()