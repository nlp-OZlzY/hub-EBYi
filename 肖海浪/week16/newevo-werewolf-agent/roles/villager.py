"""村民角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
夜间行动：无（整晚睡觉）
"""

from typing import Dict, Any
from roles.base import BaseRole, RoleType, Camp


class Villager(BaseRole):
    """村民角色

    好人阵营的普通成员，没有特殊能力。
    只能通过白天发言和投票来帮助好人阵营获胜。
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.VILLAGER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def is_night_actionable(self) -> bool:
        """村民没有夜间行动能力"""
        return False

    def get_private_context(self) -> Dict[str, Any]:
        """村民的私有上下文"""
        ctx = super().get_private_context()
        ctx["note"] = "你是村民，没有特殊能力。通过分析发言找出狼人。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        """好人阵营胜利条件：所有狼人被消灭"""
        players = game_state.get("players", [])

        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]

        return len(alive_wolves) == 0