"""配置管理"""
import os
from dataclasses import dataclass, field
from typing import Optional

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class GameConfig:
    """游戏配置"""
    player_names: list[str] = field(default_factory=list)
    role_config: list = field(default_factory=list)
    speech_rounds: int = 1
    speech_max_length: int = 200

    def __post_init__(self):
        if not self.player_names:
            self.player_names = [
                "玩家1", "玩家2", "玩家3", "玩家4", "玩家5", "玩家6",
                "玩家7", "玩家8", "玩家9", "玩家10", "玩家11", "玩家12"
            ]


@dataclass
class Config:
    """全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    game: GameConfig = field(default_factory=GameConfig)

    @classmethod
    def load(cls) -> "Config":
        """从环境变量加载配置"""
        llm_config = LLMConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )

        return cls(llm=llm_config)


# 全局配置实例
config = Config.load()