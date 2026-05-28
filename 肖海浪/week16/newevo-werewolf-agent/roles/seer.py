"""预言家角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
夜间行动：每晚查验一名玩家，返回"好人"或"狼人"
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Seer(BaseRole):
    """预言家角色

    好人阵营的核心神职，每晚可查验一名玩家身份。
    查验结果只有预言家本人知道，需要通过发言说服好人。
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.SEER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def is_night_actionable(self) -> bool:
        """预言家拥有夜间查验能力"""
        return self._is_alive

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        """获取夜间查验行动

        实际目标由 PlayerAgent 的 LLM 决策，此处返回空目标的行动模板。
        """
        return NightAction(
            action_type="check",
            target=None,  # 由 AI 代理决定查验目标
            metadata={}
        )

    def get_private_context(self) -> Dict[str, Any]:
        """预言家的私有上下文"""
        ctx = super().get_private_context()
        ctx["note"] = "你是预言家。每晚可以查验一名玩家是好人还是狼人。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        """好人阵营胜利条件：所有狼人被消灭"""
        players = game_state.get("players", [])

        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]

        return len(alive_wolves) == 0