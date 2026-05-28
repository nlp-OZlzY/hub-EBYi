from typing import Optional, List, Dict, Any
from .base_role import RolePlugin
from ..types import Player, GameState, Action, ActionType, GamePhase, RoleType

class WerewolfPlugin(RolePlugin):
    def get_role_name(self) -> str:
        return "狼人"
    
    def on_night_start(self, state: GameState) -> Optional[Action]:
        wolves = state.get_wolves()
        targets = [p for p in state.get_alive_players() 
                   if p.role != RoleType.WEREWOLF]
        
        if not targets:
            return None
        
        target = targets[0].player_id
        
        return Action(
            type=ActionType.SKILL,
            target=target,
            content="刀人"
        )
    
    def on_speech_phase(self, state: GameState) -> str:
        return f"我是{self.player.name}，我是一个普通村民。昨晚没有任何信息。我觉得今天应该先听其他人怎么说。"
    
    def get_valid_actions(self, phase: GamePhase) -> List[str]:
        if phase == GamePhase.NIGHT_WOLF:
            return ["刀人"]
        elif phase == GamePhase.SPEECH:
            return ["发言"]
        elif phase == GamePhase.VOTE:
            return ["投票"]
        return []
    
    def win_condition(self, state: GameState) -> bool:
        roles = state.get_roles()
        civilians = state.get_civilians()
        return len(roles) == 0 or len(civilians) == 0
