"""狼人 Agent"""
from typing import Optional
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState, NightAction
from core.message_bus import MessageBus
from agents.base_agent import BaseAgent
from llm.base import BaseLLM


class WerewolfAgent(BaseAgent):
    """狼人 Agent

    狼人策略：
    1. 隐藏身份，装作好人
    2. 夜间协调刀人目标
    3. 白天引导舆论，陷害好人
    4. 保护同伴狼人
    5. 投票给非狼人玩家
    """

    def __init__(self, agent_id: str, name: str, llm: BaseLLM, role: Role):
        super().__init__(agent_id, name, llm, role)
        self.wolf_companions: list[str] = []
        self.target_history: list[str] = []
        self.blamed_players: list[str] = []

    def set_companions(self, companions: list[str]):
        """设置狼人同伴"""
        self.wolf_companions = companions

    def act(self, game_state: GameState, context: dict) -> dict:
        """执行狼人行动"""
        self_understanding = self.understand_self(game_state)

        if game_state.phase.value == "night":
            return self._act_night(game_state, context)
        elif game_state.phase.value == "day_speech":
            return self._act_speech(game_state, context)
        elif game_state.phase.value == "day_vote":
            return self._act_vote(game_state, context)

        return {"action": "skip"}

    def _act_night(self, game_state: GameState, context: dict) -> dict:
        """夜间行动 - 刀人"""
        alive_players = [p for p in game_state.players.values() if p.alive]
        good_players = [
            p for p in alive_players
            if p.role.camp == Camp.GOOD and p.id != self.agent_id
        ]

        # 分析应该刀谁
        target = self._choose_kill_target(good_players, game_state, context)

        if target:
            self.target_history.append(target)
            return {
                "action": "kill",
                "target": target,
                "target_name": game_state.players[target].name,
                "reasoning": f"选择 {game_state.players[target].name} 作为刀人目标"
            }

        return {"action": "skip"}

    def _choose_kill_target(self, good_players: list, game_state: GameState, context: dict) -> Optional[str]:
        """选择刀人目标"""
        # 优先刀关键角色
        priority_targets = {
            RoleType.SEER: 3,
            RoleType.WITCH: 2,
            RoleType.GUARD: 2,
            RoleType.HUNTER: 1,
            RoleType.VILLAGER: 0
        }

        scored_targets = []
        for player in good_players:
            score = priority_targets.get(player.role.role_type, 0)

            # 避免连续刀同一人
            if player.id in self.target_history[-2:]:
                score -= 2

            # 优先刀被怀疑多的（可能是神职自证身份）
            info_processor = context.get("info_processor")
            if info_processor:
                suspicion_score = info_processor.get_trust_score(player.id)
                if suspicion_score < 0:  # 被怀疑多
                    score += 1

            scored_targets.append((player.id, score))

        scored_targets.sort(key=lambda x: x[1], reverse=True)
        return scored_targets[0][0] if scored_targets else None

    def _act_speech(self, game_state: GameState, context: dict) -> dict:
        """白天发言 - 伪装好人"""
        alive = [p for p in game_state.players.values() if p.alive]
        speeches = context.get("public_info", {}).get("public_speeches", [])

        # 分析局势，决定发言策略
        strategy = self._analyze_speech_strategy(speeches, game_state)

        # 选择陷害目标（不是狼人的玩家）
        blame_target = self._choose_blame_target(game_state, context)

        # 构建发言
        speech_prompt = f"""你是 {self.name}，在狼人杀游戏中。
你实际上是狼人，但需要伪装成好人发言。

当前局势：
- 存活玩家：{len(alive)}
- 狼人：{len([p for p in alive if p.role.camp == Camp.WOLF])} 人
- 好人：{len([p for p in alive if p.role.camp == Camp.GOOD])} 人

你的发言策略：{strategy}

{"你要陷害的目标：" + game_state.players[blame_target].name if blame_target else "暂不陷害他人"}

请生成一段自然的发言（100字左右），不要暴露狼人身份。可以适当质疑你觉得可疑的人。
"""

        speech = self.llm.chat([{"role": "user", "content": speech_prompt}])

        return {
            "action": "speech",
            "content": speech,
            "strategy": strategy,
            "blamed": blame_target
        }

    def _choose_blame_target(self, game_state: GameState, context: dict) -> Optional[str]:
        """选择陷害目标"""
        alive = [p for p in game_state.players.values()
                if p.alive and p.role.camp == Camp.GOOD and p.id not in self.wolf_companions]

        if not alive:
            return None

        # 选择被怀疑最多的好人（非狼人）
        info_processor = context.get("info_processor")
        if info_processor:
            for player in alive:
                if info_processor.get_trust_score(player.id) < 0:
                    return player.id

        # 或者选择没有被投票过的
        for player in alive:
            if info_processor and info_processor.get_vote_target_count(player.id) == 0:
                return player.id

        return alive[0].id

    def _analyze_speech_strategy(self, speeches: list, game_state: GameState) -> str:
        """分析发言策略"""
        # 检查是否有人怀疑自己
        for speech in speeches:
            if self.name in speech.get("content", ""):
                return "防守"
            # 检查是否有狼人同伴被怀疑
            for companion_id in self.wolf_companions:
                companion = game_state.players.get(companion_id)
                if companion and companion.name in speech.get("content", ""):
                    return "转移"

        return "观察"

    def _act_vote(self, game_state: GameState, context: dict) -> dict:
        """投票 - 投给非狼人玩家"""
        alive = [p for p in game_state.players.values()
                if p.alive and p.role.camp == Camp.GOOD and p.id not in self.wolf_companions]

        if not alive:
            return {"action": "skip"}

        # 使用信息处理器选择投票目标
        info_processor = context.get("info_processor")
        if info_processor:
            target = info_processor.get_most_suspicious(exclude=self.wolf_companions + [self.agent_id])
            if target and target in [p.id for p in alive]:
                return {
                    "action": "vote",
                    "target": target,
                    "target_name": game_state.players[target].name
                }

        # 默认投给第一个好人
        target = alive[0]
        return {
            "action": "vote",
            "target": target.id,
            "target_name": target.name
        }