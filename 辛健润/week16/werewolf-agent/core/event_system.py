"""事件系统"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime


class EventType(Enum):
    """事件类型"""
    GAME_START = "game_start"
    NIGHT_START = "night_start"
    NIGHT_END = "night_end"
    DAY_START = "day_start"
    DAY_END = "day_end"
    PLAYER_DEATH = "player_death"
    VOTE_START = "vote_start"
    VOTE_END = "vote_end"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    WEREWOLF_KILL = "werewolf_kill"
    SEER_REVEAL = "seer_reveal"
    WITCH_HEAL = "witch_heal"
    WITCH_POISON = "witch_poison"
    GUARD_PROTECT = "guard_protect"
    GAME_OVER = "game_over"


@dataclass
class Event:
    """事件"""
    type: EventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict = field(default_factory=dict)
    source: Optional[str] = None


class EventSystem:
    """事件系统 - 发布订阅模式"""

    def __init__(self):
        self.listeners: dict[EventType, list[Callable]] = {}
        self.event_log: list[Event] = []

    def subscribe(self, event_type: EventType, callback: Callable):
        """订阅事件"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """取消订阅"""
        if event_type in self.listeners:
            self.listeners[event_type].remove(callback)

    def publish(self, event: Event):
        """发布事件"""
        self.event_log.append(event)

        if event.type in self.listeners:
            for callback in self.listeners[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Event callback error: {e}")

    def get_event_log(self) -> list[Event]:
        """获取事件日志"""
        return self.event_log

    def clear(self):
        """清空事件日志"""
        self.event_log.clear()