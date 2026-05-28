from ..types import GameState, Team, RoleType

def check_win_condition(state: GameState) -> bool:
    alive_players = state.get_alive_players()
    
    if not alive_players:
        return True
    
    wolves_alive = any(p.role == RoleType.WEREWOLF for p in alive_players)
    good_alive = any(p.team == Team.GOOD for p in alive_players)
    
    if not wolves_alive:
        state.winner = Team.GOOD
        state.is_game_over = True
        return True
    
    if not good_alive:
        state.winner = Team.EVIL
        state.is_game_over = True
        return True
    
    roles_alive = any(p.role in [RoleType.SEER, RoleType.WITCH, RoleType.HUNTER] 
                      for p in alive_players)
    civilians_alive = any(p.role == RoleType.VILLAGER for p in alive_players)
    
    if not roles_alive:
        state.winner = Team.EVIL
        state.is_game_over = True
        return True
    
    if not civilians_alive:
        state.winner = Team.EVIL
        state.is_game_over = True
        return True
    
    return False
