from enum import Enum
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel

class RoleType(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"

class Team(str, Enum):
    GOOD = "good"
    EVIL = "evil"

class GamePhase(str, Enum):
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH = "night_witch"
    NIGHT_RESULT = "night_result"
    DAY_START = "day_start"
    SPEECH = "speech"
    VOTE = "vote"
    DAY_END = "day_end"
    GAME_OVER = "game_over"

class ActionType(str, Enum):
    SPEAK = "speak"
    VOTE = "vote"
    SKILL = "skill"
    PASS = "pass"

class Player(BaseModel):
    player_id: int
    name: str
    role: RoleType
    team: Team
    is_alive: bool = True
    has_skill: bool = True
    is_sheriff: bool = False
    
class Action(BaseModel):
    type: ActionType
    target: Optional[int] = None
    content: Optional[str] = None

class GameAction(BaseModel):
    inner_monologue: str
    action: Action
    belief_update: Dict[int, float] = {}

class GameState(BaseModel):
    game_id: str
    phase: GamePhase
    day_number: int = 1
    players: List[Player]
    dialogues: List[Dict[str, Any]] = []
    deaths: List[int] = []
    is_game_over: bool = False
    winner: Optional[Team] = None
    step_data: Dict[str, Any] = {}
    current_speaker: Optional[int] = None
    votes: Dict[int, int] = {}
    
    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]
    
    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None
    
    def get_wolves(self) -> List[Player]:
        return [p for p in self.players if p.role == RoleType.WEREWOLF and p.is_alive]
    
    def get_good_players(self) -> List[Player]:
        return [p for p in self.players if p.team == Team.GOOD and p.is_alive]
    
    def get_civilians(self) -> List[Player]:
        civilians = [RoleType.VILLAGER]
        return [p for p in self.players if p.role in civilians and p.is_alive]
    
    def get_roles(self) -> List[Player]:
        roles = [RoleType.SEER, RoleType.WITCH, RoleType.HUNTER]
        return [p for p in self.players if p.role in roles and p.is_alive]

class GameConfig(BaseModel):
    name: str
    total_players: int
    werewolf_count: int
    seer_count: int = 0
    witch_count: int = 0
    hunter_count: int = 0
    villager_count: int = 0
