import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from engine.types import GameState, Player

class BaseStrategy(ABC):
    def __init__(self, style: str = "balanced"):
        self.style = style
        self.risk_tolerance = self._get_risk_tolerance()
    
    def _get_risk_tolerance(self) -> float:
        if self.style == "bold":
            return 0.8
        elif self.style == "cautious":
            return 0.2
        else:
            return 0.5
    
    @abstractmethod
    def think(self, context: Dict[str, Any], beliefs: Dict[int, float]) -> str:
        pass
    
    @abstractmethod
    def generate_speech(self, state: GameState, player: Player) -> str:
        pass
