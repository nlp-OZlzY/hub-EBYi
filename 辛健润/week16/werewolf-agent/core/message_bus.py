"""消息总线 - 实现信息隔离与分发"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from core.role_system import RoleType, Camp


class MessageType(Enum):
    """消息类型"""
    PUBLIC_SPEECH = "public_speech"      # 公开发言
    PRIVATE_ACTION = "private_action"   # 私密行动
    VOTE = "vote"                       # 投票
    DEATH = "death"                     # 死亡信息
    GAME_OVER = "game_over"             # 游戏结束


@dataclass
class Message:
    """消息"""
    type: MessageType
    sender: str                           # 发送者ID
    content: str                          # 内容
    recipient: Optional[str] = None       # 接收者（None表示广播）
    meta: dict = field(default_factory=dict)


class MessageBus:
    """消息总线 - 控制信息分发"""

    def __init__(self):
        self.messages: list[Message] = []
        self.private_info: dict[str, list[Message]] = {}  # 私密信息

    def broadcast(self, sender: str, content: str, msg_type: MessageType = MessageType.PUBLIC_SPEECH):
        """广播消息"""
        msg = Message(
            type=msg_type,
            sender=sender,
            content=content
        )
        self.messages.append(msg)
        return msg

    def whisper(self, sender: str, recipient: str, content: str,
               msg_type: MessageType = MessageType.PRIVATE_ACTION):
        """私密消息"""
        msg = Message(
            type=msg_type,
            sender=sender,
            content=content,
            recipient=recipient
        )
        self.messages.append(msg)
        # 存储私密信息
        if recipient not in self.private_info:
            self.private_info[recipient] = []
        self.private_info[recipient].append(msg)
        return msg

    def get_public_info(self, player_id: str, role: RoleType, camp: Camp) -> dict:
        """获取公开信息（考虑信息隔离）"""
        public_messages = [m for m in self.messages if m.recipient is None]

        result = {
            "public_speeches": [],
            "votes": [],
            "deaths": [],
            "game_over": None
        }

        for msg in public_messages:
            if msg.type == MessageType.PUBLIC_SPEECH:
                result["public_speeches"].append({
                    "sender": msg.sender,
                    "content": msg.content
                })
            elif msg.type == MessageType.VOTE:
                result["votes"].append({
                    "sender": msg.sender,
                    "target": msg.meta.get("target"),
                    "content": msg.content
                })
            elif msg.type == MessageType.DEATH:
                result["deaths"].append({
                    "player": msg.meta.get("player"),
                    "reason": msg.meta.get("reason"),
                    "content": msg.content
                })
            elif msg.type == MessageType.GAME_OVER:
                result["game_over"] = {
                    "winner": msg.meta.get("winner"),
                    "content": msg.content
                }

        return result

    def get_private_info(self, player_id: str, role: RoleType, camp: Camp) -> dict:
        """获取私有信息（角色特殊能力相关信息）"""
        private_messages = []

        # 自己的私密信息
        if player_id in self.private_info:
            private_messages.extend(self.private_info[player_id])

        result = {
            "private_messages": private_messages,
            "role_specific": {}
        }

        # 根据角色添加特殊信息
        if role == RoleType.WEREWOLF:
            # 狼人知道其他狼人
            result["role_specific"]["wolves"] = []  # 由GameEngine注入
        elif role == RoleType.SEER:
            # 预言家验人结果
            result["role_specific"]["verified_players"] = []
        elif role == RoleType.WITCH:
            # 女巫的解药和毒药状态
            result["role_specific"]["potion"] = {"heal": True, "poison": True}
        elif role == RoleType.GUARD:
            # 守卫的守护记录
            result["role_specific"]["guard_history"] = []
        elif role == RoleType.HUNTER:
            # 猎人的枪状态
            result["role_specific"]["can_shoot"] = True

        return result

    def get_all_messages(self) -> list[Message]:
        """获取所有消息（用于日志）"""
        return self.messages