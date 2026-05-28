from typing import Optional, List, Dict, Any
from .base_role import RolePlugin
from ..types import Player, GameState, Action, ActionType, GamePhase

class HunterPlugin(RolePlugin):
    def __init__(self, player: Player):
        super().__init__(player)
        self.can_shoot = True
    
    def get_role_name(self) -> str:
        return "猎人"
    
    def on_night_start(self, state: GameState) -> Optional[Action]:
        return None
    
    def on_speech_phase(self, state: GameState) -> str:
        return f"我是{self.player.name}，我是猎人。如果我被狼人刀死或者被投票出局，我会开枪带走一个人！"
    
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
    
    def on_death(self, state: GameState, is_poisoned: bool = False) -> Optional[Action]:
        if not is_poisoned and self.can_shoot:
            targets = [p for p in state.get_alive_players() 
                       if p.player_id != self.player.player_id]
            if targets:
                self.can_shoot = False
                return Action(
                    type=ActionType.SKILL,
                    target=targets[0].player_id,
                    content="开枪带走"
                )
        return None
