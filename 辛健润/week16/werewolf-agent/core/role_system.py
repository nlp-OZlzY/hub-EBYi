"""角色系统定义"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RoleType(Enum):
    """角色类型"""
    WEREWOLF = "werewolf"       # 狼人
    SEER = "seer"               # 预言家
    WITCH = "witch"             # 女巫
    GUARD = "guard"             # 守卫
    VILLAGER = "villager"       # 村民
    HUNTER = "hunter"           # 猎人


class Camp(Enum):
    """阵营"""
    GOOD = "good"               # 好人
    WOLF = "wolf"               # 狼人


@dataclass
class Role:
    """角色定义"""
    role_type: RoleType
    name: str
    camp: Camp
    abilities: list[str]
    description: str

    @staticmethod
    def get_role_config(role_type: RoleType) -> "Role":
        """获取角色配置"""
        configs = {
            RoleType.WEREWOLF: Role(
                role_type=RoleType.WEREWOLF,
                name="狼人",
                camp=Camp.WOLF,
                abilities=["刀人", "夜间交流"],
                description="每晚可以杀死一名玩家，知道其他狼人同伴"
            ),
            RoleType.SEER: Role(
                role_type=RoleType.SEER,
                name="预言家",
                camp=Camp.GOOD,
                abilities=["验人"],
                description="每晚可以查验一名玩家的阵营"
            ),
            RoleType.WITCH: Role(
                role_type=RoleType.WITCH,
                name="女巫",
                camp=Camp.GOOD,
                abilities=["救人", "毒人"],
                description="有一瓶解药和一瓶毒药，每晚可以选择使用"
            ),
            RoleType.GUARD: Role(
                role_type=RoleType.GUARD,
                name="守卫",
                camp=Camp.GOOD,
                abilities=["守人"],
                description="每晚可以守护一名玩家，不能连续守同一人"
            ),
            RoleType.VILLAGER: Role(
                role_type=RoleType.VILLAGER,
                name="村民",
                camp=Camp.GOOD,
                abilities=[],
                description="普通村民，没有任何特殊能力"
            ),
            RoleType.HUNTER: Role(
                role_type=RoleType.HUNTER,
                name="猎人",
                camp=Camp.GOOD,
                abilities=["开枪"],
                description="死亡时可以开枪带走一人"
            ),
        }
        return configs[role_type]


# 标准12人局配置
STANDARD_12_PLAYER_CONFIG = [
    RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF,  # 3狼
    RoleType.SEER,                                             # 预言家
    RoleType.WITCH,                                            # 女巫
    RoleType.GUARD,                                            # 守卫
    RoleType.HUNTER,                                           # 猎人
    RoleType.VILLAGER, RoleType.VILLAGER,                      # 2村民
    RoleType.VILLAGER, RoleType.VILLAGER,                      # 3村民（凑12人）
    RoleType.VILLAGER,
]