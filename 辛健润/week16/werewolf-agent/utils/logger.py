"""结构化日志工具"""
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class GameLog:
    """游戏日志"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    level: LogLevel = LogLevel.INFO
    phase: str = ""
    event: str = ""
    player: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "phase": self.phase,
            "event": self.event,
            "player": self.player,
            "content": self.content,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class GameLogger:
    """游戏日志记录器"""

    def __init__(self):
        self.logs: list[GameLog] = []

    def log(
        self,
        event: str,
        content: str = "",
        level: LogLevel = LogLevel.INFO,
        phase: str = "",
        player: str = "",
        **metadata
    ):
        """记录日志"""
        log = GameLog(
            timestamp=datetime.now().isoformat(),
            level=level,
            phase=phase,
            event=event,
            player=player,
            content=content,
            metadata=metadata
        )
        self.logs.append(log)

    def info(self, event: str, content: str = "", **kwargs):
        self.log(event, content, LogLevel.INFO, **kwargs)

    def warning(self, event: str, content: str = "", **kwargs):
        self.log(event, content, LogLevel.WARNING, **kwargs)

    def error(self, event: str, content: str = "", **kwargs):
        self.log(event, content, LogLevel.ERROR, **kwargs)

    def debug(self, event: str, content: str = "", **kwargs):
        self.log(event, content, LogLevel.DEBUG, **kwargs)

    def get_logs(self) -> list[dict]:
        """获取所有日志"""
        return [log.to_dict() for log in self.logs]

    def export_json(self) -> str:
        """导出 JSON 格式日志"""
        return json.dumps(self.get_logs(), ensure_ascii=False, indent=2)

    def print_summary(self):
        """打印日志摘要"""
        print("\n" + "=" * 60)
        print("游戏日志摘要")
        print("=" * 60)

        for log in self.logs:
            prefix = f"[{log.level.value}]"
            player_info = f"[{log.player}]" if log.player else ""
            print(f"{prefix} {player_info} {log.event}: {log.content}")

        print("=" * 60 + "\n")


# 全局日志实例
logger = GameLogger()