from statemachine import State, StateMachine
from .types import GamePhase

class GameStateMachine(StateMachine):
    night_wolf = State("night_wolf", initial=True)
    night_seer = State("night_seer")
    night_witch = State("night_witch")
    night_result = State("night_result")
    day_start = State("day_start")
    speech = State("speech")
    vote = State("vote")
    day_end = State("day_end")
    game_over = State("game_over", final=True)
    
    wolf_action = night_wolf.to(night_seer)
    seer_action = night_seer.to(night_witch)
    witch_action = night_witch.to(night_result)
    night_complete = night_result.to(day_start)
    start_day = day_start.to(speech)
    speech_complete = speech.to(vote)
    vote_complete = vote.to(day_end)
    day_complete = day_end.to(night_wolf)
    game_finish = day_end.to(game_over) | speech.to(game_over) | vote.to(game_over)
    
    def __init__(self):
        super().__init__()
        self.day_number = 1
        self.is_first_day = True
    
    def get_current_phase(self) -> GamePhase:
        return GamePhase(self.current_state.id)
    
    def advance(self, check_game_over: bool = False) -> GamePhase:
        transitions = [
            ('wolf_action', 'seer_action', 'witch_action', 'night_complete', 
             'start_day', 'speech_complete', 'vote_complete', 'day_complete')
        ]
        
        for transition_group in transitions:
            for transition in transition_group:
                if getattr(self, f'can_{transition}')():
                    getattr(self, transition)()
                    if self.get_current_phase() == GamePhase.DAY_END:
                        self.day_number += 1
                        self.is_first_day = False
                    return self.get_current_phase()
        
        if check_game_over and self.can_game_finish():
            self.game_finish()
        
        return self.get_current_phase()
    
    def reset(self):
        self.day_number = 1
        self.is_first_day = True
        self.to_night_wolf()
