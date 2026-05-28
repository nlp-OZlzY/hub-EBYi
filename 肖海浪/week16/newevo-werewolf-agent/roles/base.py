"""游戏角色基类。

所有角色都继承自此类，包含：
- 角色类型和阵营归属
- 夜间行动能力（如有）
- 白天发言和投票能力
- 死亡触发技能（如猎人开枪）
- 胜利条件检查
- 私有上下文（用于 AI 代理的 prompt 构建）
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class RoleType(Enum):
    """角色类型枚举"""
    WEREWOLF = "werewolf"   # 狼人
    SEER = "seer"           # 预言家
    WITCH = "witch"         # 女巫
    HUNTER = "hunter"       # 猎人
    VILLAGER = "villager"   # 村民


class Camp(Enum):
    """阵营枚举"""
    GOOD = "good"   # 好人阵营（神职 + 村民）
    EVIL = "evil"   # 邪恶阵营（狼人）


@dataclass
class NightAction:
    """夜间行动数据类"""
    action_type: str                          # 行动类型（kill/check/heal/poison）
    target: Optional[int] = None              # 目标玩家 ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息


@dataclass
class VoteAction:
    """投票行动数据类"""
    target: int  # 目标玩家 ID


class BaseRole:
    """游戏角色基类

    每个角色都有：角色类型、阵营、存活状态、警长身份。
    子类需实现 role_type 和 camp 属性，以及按需覆盖夜间行动和胜利条件。
    """

    def __init__(self, player_id: int):
        """初始化角色

        Args:
            player_id: 所属玩家的 ID
        """
        self.player_id = player_id
        self._is_alive = True
        self._is_sheriff = False

    @property
    def role_type(self) -> RoleType:
        """返回角色类型，子类必须实现"""
        raise NotImplementedError

    @property
    def camp(self) -> Camp:
        """返回所属阵营，子类必须实现"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """返回角色显示名称"""
        return self.role_type.value.capitalize()

    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value: bool):
        self._is_alive = value

    @property
    def is_sheriff(self) -> bool:
        """是否为警长"""
        return self._is_sheriff

    @is_sheriff.setter
    def is_sheriff(self, value: bool):
        self._is_sheriff = value

    def is_night_actionable(self) -> bool:
        """是否拥有夜间行动能力"""
        return False

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        """获取夜间行动决策

        Args:
            game_state: 当前游戏状态

        Returns:
            NightAction 如果要行动，None 表示不行动
        """
        return None

    def can_speak(self) -> bool:
        """是否能在白天发言"""
        return self._is_alive

    def get_speech(self, game_state: Dict[str, Any]) -> str:
        """生成白天发言内容

        Args:
            game_state: 当前游戏状态

        Returns:
            发言文本
        """
        return ""

    def can_vote(self) -> bool:
        """是否能投票"""
        return self._is_alive

    def get_vote_target(self, game_state: Dict[str, Any]) -> Optional[VoteAction]:
        """获取投票目标

        Args:
            game_state: 当前游戏状态

        Returns:
            VoteAction 如果要投票，None 表示弃票
        """
        return None

    def on_death(self, cause: str, game_state: Dict[str, Any]) -> Optional[List[int]]:
        """角色死亡时触发

        Args:
            cause: 死亡原因（night_kill/vote/shoot/poison）
            game_state: 当前游戏状态

        Returns:
            受死亡技能影响的玩家 ID 列表（如猎人开枪）
        """
        self._is_alive = False
        return None

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        """检查本阵营是否胜利

        Args:
            game_state: 当前游戏状态

        Returns:
            True 表示本阵营已胜利
        """
        raise NotImplementedError

    def get_private_context(self) -> Dict[str, Any]:
        """获取角色私有上下文（用于 AI 代理构建 prompt）

        Returns:
            包含角色名称、玩家 ID、阵营等私有信息的字典
        """
        return {
            "role": self.name,
            "player_id": self.player_id,
            "camp": self.camp.value,
        }

    def __repr__(self) -> str:
        status = "alive" if self._is_alive else "dead"
        sheriff = " (sheriff)" if self._is_sheriff else ""
        return f"{self.__class__.__name__}(player_id={self.player_id}, {status}{sheriff})"