import uuid
import logging
from typing import Dict, Optional, List
from .types import GameState, GamePhase, Player, RoleType, Team, GameConfig, GameAction, ActionType
from .state_machine import GameStateMachine
from .rules.win_conditions import check_win_condition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class GameManager:
    def __init__(self):
        self.games: Dict[str, GameState] = {}
        self.state_machines: Dict[str, GameStateMachine] = {}
    
    def create_game(self, config: GameConfig, player_names: Optional[List[str]] = None, 
                   shuffle: bool = True) -> str:
        game_id = str(uuid.uuid4())[:8]
        
        players = []
        role_order = []
        
        for _ in range(config.werewolf_count):
            role_order.append(RoleType.WEREWOLF)
        for _ in range(config.seer_count):
            role_order.append(RoleType.SEER)
        for _ in range(config.witch_count):
            role_order.append(RoleType.WITCH)
        for _ in range(config.hunter_count):
            role_order.append(RoleType.HUNTER)
        for _ in range(config.villager_count):
            role_order.append(RoleType.VILLAGER)
        
        if shuffle:
            import random
            random.shuffle(role_order)
        
        for i, role in enumerate(role_order):
            name = player_names[i] if player_names and i < len(player_names) else f"玩家{i+1}"
            team = Team.EVIL if role == RoleType.WEREWOLF else Team.GOOD
            players.append(Player(
                player_id=i,
                name=name,
                role=role,
                team=team,
                is_alive=True,
                has_skill=True
            ))
        
        state = GameState(
            game_id=game_id,
            phase=GamePhase.NIGHT_WOLF,
            day_number=1,
            players=players
        )
        
        self.games[game_id] = state
        self.state_machines[game_id] = GameStateMachine()
        
        logger.info(f"[GAME_CREATE] game_id={game_id}, config={config.name}, total_players={config.total_players}")
        for p in players:
            logger.debug(f"  - player={p.player_id}, name={p.name}, role={p.role.value}, team={p.team.value}")
        
        return game_id
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        return self.games.get(game_id)
    
    def step(self, game_id: str) -> GameState:
        state = self.games.get(game_id)
        if not state:
            logger.error(f"[GAME_STEP] Game {game_id} not found")
            raise ValueError(f"Game {game_id} not found")
        
        if state.is_game_over:
            logger.info(f"[GAME_STEP] game_id={game_id}, game already over, winner={state.winner.value if state.winner else None}")
            return state
        
        sm = self.state_machines[game_id]
        current_phase = sm.get_current_phase()
        
        logger.info(f"[GAME_STEP] game_id={game_id}, day={state.day_number}, current_phase={current_phase.value}")
        
        if current_phase == GamePhase.NIGHT_WOLF:
            self._handle_night_wolf(state)
            sm.wolf_action()
        elif current_phase == GamePhase.NIGHT_SEER:
            self._handle_night_seer(state)
            sm.seer_action()
        elif current_phase == GamePhase.NIGHT_WITCH:
            self._handle_night_witch(state)
            sm.witch_action()
        elif current_phase == GamePhase.NIGHT_RESULT:
            self._handle_night_result(state)
            sm.night_complete()
        elif current_phase == GamePhase.DAY_START:
            sm.start_day()
        elif current_phase == GamePhase.SPEECH:
            self._handle_speech(state)
            sm.speech_complete()
        elif current_phase == GamePhase.VOTE:
            self._handle_vote(state)
            sm.vote_complete()
        elif current_phase == GamePhase.DAY_END:
            self._handle_day_end(state)
            if not state.is_game_over:
                sm.day_complete()
            else:
                sm.game_finish()
        
        state.phase = sm.get_current_phase()
        state.day_number = sm.day_number
        
        logger.debug(f"[GAME_STEP] game_id={game_id}, next_phase={state.phase.value}, alive_count={len(state.get_alive_players())}")
        
        return state
    
    def _handle_night_wolf(self, state: GameState):
        wolves = state.get_wolves()
        if wolves:
            wolf_votes = []
            target = wolves[0].player_id + 1 if wolves[0].player_id + 1 < len(state.players) else 0
            
            for wolf in wolves:
                wolf_votes.append({"player_id": wolf.player_id, "target": target})
            
            state.step_data = {
                "wolf_votes": wolf_votes,
                "final_target": target
            }
            
            logger.info(f"[WOLF_ACTION] game_id={state.game_id}, wolves={[w.player_id for w in wolves]}, target={target}")
    
    def _handle_night_seer(self, state: GameState):
        seers = [p for p in state.players if p.role == RoleType.SEER and p.is_alive]
        if seers:
            seer = seers[0]
            targets = [p for p in state.players if p.is_alive and p.player_id != seer.player_id]
            if targets:
                target = targets[0].player_id
                target_player = state.get_player_by_id(target)
                result = "狼人" if target_player.role == RoleType.WEREWOLF else "好人"
                
                state.step_data = {
                    "seer_id": seer.player_id,
                    "target": target,
                    "result": result
                }
                
                logger.info(f"[SEER_ACTION] game_id={state.game_id}, seer={seer.player_id}, target={target}, result={result}")
    
    def _handle_night_witch(self, state: GameState):
        witches = [p for p in state.players if p.role == RoleType.WITCH and p.is_alive]
        if witches:
            witch = witches[0]
            night_target = state.step_data.get("final_target")
            
            state.step_data["witch_action"] = {
                "witch_id": witch.player_id,
                "save_target": None,
                "poison_target": None,
                "message": "女巫选择不使用药水"
            }
    
    def _handle_night_result(self, state: GameState):
        night_target = state.step_data.get("final_target")
        witch_action = state.step_data.get("witch_action", {})
        
        deaths = []
        if night_target is not None:
            if witch_action.get("save_target") != night_target:
                deaths.append(night_target)
        
        if witch_action.get("poison_target"):
            deaths.append(witch_action["poison_target"])
        
        for target_id in deaths:
            player = state.get_player_by_id(target_id)
            if player:
                player.is_alive = False
        
        state.deaths = deaths
        state.dialogues.append({
            "phase": "night_result",
            "deaths": deaths,
            "message": f"昨晚死亡的玩家: {', '.join(str(d) for d in deaths)}"
        })
    
    def _handle_speech(self, state: GameState):
        alive_players = state.get_alive_players()
        for player in alive_players:
            speech_content = f"{player.name} (身份未知): 我是好人，昨晚没有特殊信息。"
            state.dialogues.append({
                "phase": "speech",
                "speaker_id": player.player_id,
                "speaker_name": player.name,
                "content": speech_content
            })
    
    def _handle_vote(self, state: GameState):
        alive_players = state.get_alive_players()
        votes = {}
        
        if alive_players:
            vote_target = alive_players[-1].player_id
            
            for player in alive_players:
                votes[player.player_id] = vote_target
            
            vote_counts = {}
            for voter, target in votes.items():
                vote_counts[target] = vote_counts.get(target, 0) + 1
            
            max_votes = max(vote_counts.values()) if vote_counts else 0
            targets_with_max = [t for t, c in vote_counts.items() if c == max_votes]
            
            if targets_with_max:
                eliminated = targets_with_max[0]
                eliminated_player = state.get_player_by_id(eliminated)
                
                if eliminated_player:
                    eliminated_player.is_alive = False
                    state.deaths.append(eliminated)
                    state.dialogues.append({
                        "phase": "vote",
                        "votes": votes,
                        "eliminated": eliminated,
                        "eliminated_name": eliminated_player.name,
                        "message": f"{eliminated_player.name} 被投票出局"
                    })
                    
                    logger.info(f"[VOTE_RESULT] game_id={state.game_id}, votes={votes}, eliminated={eliminated}, eliminated_name={eliminated_player.name}")
        
        state.votes = votes
    
    def _handle_day_end(self, state: GameState):
        check_win_condition(state)
        
        if state.is_game_over:
            winner_text = "好人阵营" if state.winner == Team.GOOD else "狼人阵营"
            state.dialogues.append({
                "phase": "game_over",
                "winner": state.winner.value,
                "message": f"游戏结束！{winner_text}获胜！"
            })
            
            logger.info(f"[GAME_END] game_id={state.game_id}, winner={state.winner.value}, day={state.day_number}, alive_count={len(state.get_alive_players())}")
    
    def delete_game(self, game_id: str):
        if game_id in self.games:
            del self.games[game_id]
            del self.state_machines[game_id]
    
    def list_games(self) -> List[Dict[str, any]]:
        result = []
        for game_id, state in self.games.items():
            result.append({
                "game_id": game_id,
                "phase": state.phase.value,
                "day_number": state.day_number,
                "alive_count": len(state.get_alive_players()),
                "is_game_over": state.is_game_over,
                "winner": state.winner.value if state.winner else None
            })
        return result
