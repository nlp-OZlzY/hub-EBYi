"""女巫角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
夜间行动：
    - 有一瓶救人药（复活今晚死亡的人）和一瓶毒药（杀死任意玩家）
    - 各只能使用一次
    - 不能救自己
    - 同一晚上不能同时使用
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Witch(BaseRole):
    """女巫角色

    好人阵营的核心神职，拥有一瓶解药和一瓶毒药：
    - 解药：救活当晚被狼人击杀的玩家（不能自救）
    - 毒药：毒死任意一名玩家
    - 单夜不能同时使用两瓶药
    """

    def __init__(self, player_id: int):
        """初始化女巫，两瓶药初始可用，用后不可恢复"""
        super().__init__(player_id)
        self._has_heal = True    # 解药是否可用（初始可用）
        self._has_poison = True  # 毒药是否可用（初始可用）
        self._used_heal = False  # 解药是否已使用（使用后锁定）
        self._used_poison = False  # 毒药是否已使用（使用后锁定）

    @property
    def role_type(self) -> RoleType:
        return RoleType.WITCH

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    @property
    def has_heal(self) -> bool:
        """解药是否可用（未使用过）"""
        return self._has_heal and not self._used_heal

    @property
    def has_poison(self) -> bool:
        """毒药是否可用（未使用过）"""
        return self._has_poison and not self._used_poison

    def is_night_actionable(self) -> bool:
        """女巫在还有药可用时拥有夜间行动能力"""
        return self._is_alive and (self.has_heal or self.has_poison)

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        """获取夜间用药行动

        女巫可以看到今晚的死亡玩家，决定是否救人或使用毒药。
        """
        tonight_death = game_state.get("tonight_death")  # 今晚被击杀的玩家 ID

        return NightAction(
            action_type="witch",
            target=None,
            metadata={
                "has_heal": self.has_heal,
                "has_poison": self.has_poison,
                "tonight_death": tonight_death,
                "can_heal_self": tonight_death != self.player_id if tonight_death else True,
            }
        )

    def use_heal(self):
        """标记解药已使用"""
        self._used_heal = True

    def use_poison(self):
        """标记毒药已使用"""
        self._used_poison = True

    def get_private_context(self) -> Dict[str, Any]:
        """女巫的私有上下文：包含药水状态"""
        ctx = super().get_private_context()
        ctx.update({
            "has_heal": self.has_heal,
            "has_poison": self.has_poison,
            "note": "你是女巫。你有一瓶解药（救活今晚死者）和一瓶毒药（毒死任意玩家），各限用一次，同夜不能双用。"
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