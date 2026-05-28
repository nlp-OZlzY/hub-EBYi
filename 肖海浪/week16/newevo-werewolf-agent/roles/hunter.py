"""猎人角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
被动技能：夜晚被杀或白天被投票出局时，可在死前开枪带走一人
特殊：如果被女巫毒杀，无法开枪（技能锁定）
"""

from typing import Optional, List, Dict, Any
from roles.base import BaseRole, RoleType, Camp


class Hunter(BaseRole):
    """猎人角色

    好人阵营的神职，拥有被动技能「枪魂」：
    - 被狼人击杀或被投票出局时，可开枪带走一名玩家
    - 被女巫毒杀时无法开枪（技能锁定）
    """

    def __init__(self, player_id: int):
        """初始化猎人，设置开枪能力状态"""
        super().__init__(player_id)
        self._can_shoot = True  # 被毒杀时锁定

    @property
    def role_type(self) -> RoleType:
        return RoleType.HUNTER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    @property
    def can_shoot(self) -> bool:
        """是否可以开枪"""
        return self._can_shoot

    def lock_shoot(self):
        """锁定猎人开枪能力（被毒杀时调用）"""
        self._can_shoot = False

    def on_death(self, cause: str, game_state: Dict[str, Any]) -> Optional[List[int]]:
        """猎人死亡时触发开枪技能

        触发条件：被狼人击杀（night_kill）或被投票出局（vote）
        锁定条件：被女巫毒杀（poison）时无法开枪

        Returns:
            空列表 [] 表示要开枪（目标由 GameEngine._handle_hunter_death 决定）
            None 表示不开枪
        """
        self._is_alive = False

        # 被毒杀时无法开枪
        if cause == "poison":
            self._can_shoot = False
            return None

        # 被狼人杀或被投票出局时可以开枪
        if cause in ("night_kill", "vote") and self._can_shoot:
            return []  # 空列表表示要开枪，目标由 GameEngine 决定

        return None

    def get_shoot_target(self, game_state: Dict[str, Any]) -> int:
        """获取开枪目标（由 AI 代理决策）

        Returns:
            目标玩家 ID
        """
        return -1  # 由 AI 代理决定

    def get_private_context(self) -> Dict[str, Any]:
        """猎人的私有上下文：包含开枪能力状态"""
        ctx = super().get_private_context()
        ctx.update({
            "can_shoot": self._can_shoot,
            "note": "你是猎人。死亡时（非毒杀）可以开枪带走一名玩家。"
        })
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        """好人阵营胜利条件：所有狼人被消灭"""
        players = game_state.get("players", [])

        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]

        return len(alive_wolves) == 0