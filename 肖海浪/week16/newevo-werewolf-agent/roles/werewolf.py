"""狼人角色实现。

阵营：邪恶阵营
胜利条件：杀死所有神职 OR 杀死所有村民
夜间行动：与其他狼人协调，每晚杀死一名玩家
特殊能力：知道同伴身份，可以伪装成好人（悍跳）
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Werewolf(BaseRole):
    """狼人角色

    邪恶阵营的核心角色，夜间与同伴协商击杀目标。
    白天需要伪装成好人（悍跳），混淆视听。

    胜利条件：所有神职死亡 或 所有村民死亡
    夜间行动：与其他存活狼人投票选出击杀目标（多数票决）
    私有信息：知道所有狼人同伴的身份
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.WEREWOLF

    @property
    def camp(self) -> Camp:
        return Camp.EVIL

    def is_night_actionable(self) -> bool:
        """狼人拥有夜间击杀能力"""
        return self._is_alive

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        """获取夜间击杀行动

        狼人可以看到同伴并协调击杀目标。
        实际目标由 PlayerAgent 的 LLM 决策，此处只提供元数据。
        """
        # 查找存活的狼人同伴
        other_wolves = [
            p for p in game_state.get("players", [])
            if p.get("role") == RoleType.WEREWOLF.value
            and p.get("player_id") != self.player_id
            and p.get("is_alive", False)
        ]

        return NightAction(
            action_type="kill",
            target=None,  # 由 AI 代理决定目标
            metadata={"allies": [w["player_id"] for w in other_wolves]}
        )

    def get_private_context(self) -> Dict[str, Any]:
        """狼人的私有上下文：知道同伴身份"""
        ctx = super().get_private_context()
        ctx["knows_wolves"] = True
        ctx["note"] = "你是狼人。你知道同伴是谁，但好人不知道任何人的身份。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        """邪恶阵营胜利条件：所有神职死亡 或 所有村民死亡"""
        players = game_state.get("players", [])

        alive_gods = [p for p in players if p.get("is_god") and p.get("is_alive")]
        alive_villagers = [p for p in players if not p.get("is_god") and p.get("is_alive")]
        alive_wolves = [p for p in players if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")]

        # 狼人全灭则好人胜利
        if not alive_wolves:
            return False

        # 神职全灭或村民全灭则狼人胜利
        return len(alive_gods) == 0 or len(alive_villagers) == 0