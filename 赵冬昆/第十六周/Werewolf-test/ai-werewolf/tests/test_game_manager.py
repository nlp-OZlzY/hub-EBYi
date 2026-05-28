import pytest
from engine.game_manager import GameManager
from engine.types import GameConfig, GamePhase, RoleType, Team

class TestGameManager:
    def test_create_game(self):
        manager = GameManager()
        config = GameConfig(
            name="test_6",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = manager.create_game(config)
        state = manager.get_game(game_id)
        
        assert state is not None
        assert state.game_id == game_id
        assert state.phase == GamePhase.NIGHT_WOLF
        assert state.day_number == 1
        assert len(state.players) == 6
    
    def test_step_game(self):
        manager = GameManager()
        config = GameConfig(
            name="test_6",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = manager.create_game(config)
        
        state = manager.step(game_id)
        assert state.phase == GamePhase.NIGHT_SEER
        
        state = manager.step(game_id)
        assert state.phase == GamePhase.NIGHT_WITCH
        
        state = manager.step(game_id)
        assert state.phase == GamePhase.NIGHT_RESULT
    
    def test_win_condition_wolves_win(self):
        manager = GameManager()
        config = GameConfig(
            name="test_4",
            total_players=4,
            werewolf_count=1,
            seer_count=1,
            villager_count=2
        )
        
        game_id = manager.create_game(config)
        state = manager.get_game(game_id)
        
        for _ in range(20):
            state = manager.step(game_id)
            if state.is_game_over:
                break
        
        assert state.is_game_over
    
    def test_list_games(self):
        manager = GameManager()
        config = GameConfig(
            name="test_6",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = manager.create_game(config)
        games = manager.list_games()
        
        assert len(games) == 1
        assert games[0]["game_id"] == game_id
    
    def test_delete_game(self):
        manager = GameManager()
        config = GameConfig(
            name="test_6",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = manager.create_game(config)
        manager.delete_game(game_id)
        
        assert manager.get_game(game_id) is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
