from typing import Optional, List, Dict, Any
from .base_role import RolePlugin
from ..types import Player, GameState, Action, ActionType, GamePhase, RoleType

class SeerPlugin(RolePlugin):
    def __init__(self, player: Player):
        super().__init__(player)
        self.inspection_results = []
    
    def get_role_name(self) -> str:
        return "预言家"
    
    def on_night_start(self, state: GameState) -> Optional[Action]:
        targets = [p for p in state.get_alive_players() 
                   if p.player_id != self.player.player_id]
        
        if not targets:
            return None
        
        target = targets[0].player_id
        target_player = state.get_player_by_id(target)
        result = "狼人" if target_player.role == RoleType.WEREWOLF else "好人"
        
        self.inspection_results.append({
            "night": state.day_number,
            "target": target,
            "result": result
        })
        
        return Action(
            type=ActionType.SKILL,
            target=target,
            content=f"查验结果: {result}"
        )
    
    def on_speech_phase(self, state: GameState) -> str:
        if self.inspection_results:
            last_result = self.inspection_results[-1]
            return f"我是{self.player.name}，我是预言家。昨晚我查验了玩家{last_result['target']}，结果是{last_result['result']}！"
        return f"我是{self.player.name}，我是预言家。昨晚没有查验信息。"
    
    def get_valid_actions(self, phase: GamePhase) -> List[str]:
        if phase == GamePhase.NIGHT_SEER:
            return ["查验"]
        elif phase == GamePhase.SPEECH:
            return ["发言"]
        elif phase == GamePhase.VOTE:
            return ["投票"]
        return []
    
    def win_condition(self, state: GameState) -> bool:
        wolves = state.get_wolves()
        return len(wolves) == 0
