from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from ..types import Player, GameState, Action, GamePhase

class RolePlugin(ABC):
    def __init__(self, player: Player):
        self.player = player
    
    @abstractmethod
    def get_role_name(self) -> str:
        pass
    
    @abstractmethod
    def on_night_start(self, state: GameState) -> Optional[Action]:
        pass
    
    @abstractmethod
    def on_speech_phase(self, state: GameState) -> str:
        pass
    
    @abstractmethod
    def get_valid_actions(self, phase: GamePhase) -> List[str]:
        pass
    
    @abstractmethod
    def win_condition(self, state: GameState) -> bool:
        pass
