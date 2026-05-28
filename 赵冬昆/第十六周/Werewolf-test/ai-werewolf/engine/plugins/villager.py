from typing import Optional, List, Dict, Any
from .base_role import RolePlugin
from ..types import Player, GameState, Action, ActionType, GamePhase

class VillagerPlugin(RolePlugin):
    def get_role_name(self) -> str:
        return "村民"
    
    def on_night_start(self, state: GameState) -> Optional[Action]:
        return None
    
    def on_speech_phase(self, state: GameState) -> str:
        return f"我是{self.player.name}，我是一个普通村民。昨晚没有任何特殊信息，希望预言家能给我们一些线索。"
    
    def get_valid_actions(self, phase: GamePhase) -> List[str]:
        actions = []
        if phase == GamePhase.SPEECH:
            actions.append("发言")
        elif phase == GamePhase.VOTE:
            actions.append("投票")
        return actions
    
    def win_condition(self, state: GameState) -> bool:
        wolves = state.get_wolves()
        return len(wolves) == 0
