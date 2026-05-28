from typing import Optional, List, Dict, Any
from .base_role import RolePlugin
from ..types import Player, GameState, Action, ActionType, GamePhase

class WitchPlugin(RolePlugin):
    def __init__(self, player: Player):
        super().__init__(player)
        self.has_potion = True
        self.has_antidote = True
    
    def get_role_name(self) -> str:
        return "女巫"
    
    def on_night_start(self, state: GameState) -> Optional[Action]:
        night_target = state.step_data.get("final_target")
        
        if night_target is not None and self.has_antidote:
            self.has_antidote = False
            return Action(
                type=ActionType.SKILL,
                target=night_target,
                content="使用解药"
            )
        elif self.has_potion:
            targets = [p for p in state.get_alive_players() 
                       if p.player_id != self.player.player_id]
            if targets:
                self.has_potion = False
                return Action(
                    type=ActionType.SKILL,
                    target=targets[0].player_id,
                    content="使用毒药"
                )
        
        return Action(
            type=ActionType.PASS,
            content="不使用药水"
        )
    
    def on_speech_phase(self, state: GameState) -> str:
        status = []
        if self.has_antidote:
            status.append("解药")
        if self.has_potion:
            status.append("毒药")
        
        if status:
            return f"我是{self.player.name}，我是女巫。我还有{'、'.join(status)}。昨晚平安夜。"
        return f"我是{self.player.name}，我是女巫。我的药水都用完了。"
    
    def get_valid_actions(self, phase: GamePhase) -> List[str]:
        actions = []
        if phase == GamePhase.NIGHT_WITCH:
            if self.has_antidote:
                actions.append("使用解药")
            if self.has_potion:
                actions.append("使用毒药")
            actions.append("不使用药水")
        elif phase == GamePhase.SPEECH:
            actions.append("发言")
        elif phase == GamePhase.VOTE:
            actions.append("投票")
        return actions
    
    def win_condition(self, state: GameState) -> bool:
        wolves = state.get_wolves()
        return len(wolves) == 0
