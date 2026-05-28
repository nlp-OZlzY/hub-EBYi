import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from .base_strategy import BaseStrategy
from engine.types import GameState, Player, RoleType

class SimpleStrategy(BaseStrategy):
    def think(self, context: Dict[str, Any], beliefs: Dict[int, float]) -> str:
        role = context.get("role", "unknown")
        phase = context.get("phase", "unknown")
        
        thoughts = []
        thoughts.append(f"我是{role}，当前阶段是{phase}")
        
        if phase == "speech":
            thoughts.append("我需要发言，表明自己的立场")
        elif phase == "vote":
            thoughts.append("我需要投票，找出狼人")
        elif phase.startswith("night"):
            thoughts.append("夜晚行动时间，执行我的角色技能")
        
        return " ".join(thoughts)
    
    def generate_speech(self, state: GameState, player: Player) -> str:
        if player.role == RoleType.WEREWOLF:
            return self._generate_wolf_speech(state, player)
        elif player.role == RoleType.SEER:
            return self._generate_seer_speech(state, player)
        elif player.role == RoleType.WITCH:
            return self._generate_witch_speech(state, player)
        elif player.role == RoleType.HUNTER:
            return self._generate_hunter_speech(state, player)
        else:
            return self._generate_villager_speech(state, player)
    
    def _generate_wolf_speech(self, state: GameState, player: Player) -> str:
        return f"大家好，我是{player.name}，我是一个普通村民。昨晚没有什么特别的信息，我觉得我们应该仔细分析每个人的发言。"
    
    def _generate_seer_speech(self, state: GameState, player: Player) -> str:
        return f"我是{player.name}，我是预言家！昨晚我查验了一些信息，希望大家能相信我，跟着我的指引走。"
    
    def _generate_witch_speech(self, state: GameState, player: Player) -> str:
        return f"我是{player.name}，我是女巫。我有解药和毒药，昨晚平安夜。大家要小心狼人！"
    
    def _generate_hunter_speech(self, state: GameState, player: Player) -> str:
        return f"我是{player.name}，我是猎人。如果我被投票出局，我会开枪带走一个人！"
    
    def _generate_villager_speech(self, state: GameState, player: Player) -> str:
        return f"我是{player.name}，我是普通村民。我会根据大家的发言来判断谁是狼人。"
