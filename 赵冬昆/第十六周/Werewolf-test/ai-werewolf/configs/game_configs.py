import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.types import GameConfig

STANDARD_CONFIGS = {
    "standard_6": GameConfig(
        name="标准6人局",
        total_players=6,
        werewolf_count=2,
        seer_count=1,
        witch_count=1,
        hunter_count=0,
        villager_count=2
    ),
    "simple_4": GameConfig(
        name="简单4人局",
        total_players=4,
        werewolf_count=1,
        seer_count=1,
        witch_count=0,
        hunter_count=0,
        villager_count=2
    ),
    "big_9": GameConfig(
        name="进阶9人局",
        total_players=9,
        werewolf_count=3,
        seer_count=1,
        witch_count=1,
        hunter_count=1,
        villager_count=3
    )
}

def get_config(config_name: str) -> GameConfig:
    return STANDARD_CONFIGS.get(config_name, STANDARD_CONFIGS["standard_6"])

def list_configs() -> list:
    return [{"name": name, **config.dict()} for name, config in STANDARD_CONFIGS.items()]
