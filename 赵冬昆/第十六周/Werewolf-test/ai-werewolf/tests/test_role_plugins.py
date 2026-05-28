import pytest
from engine.types import Player, RoleType, Team, GameState, GamePhase
from engine.plugins import (
    WerewolfPlugin,
    SeerPlugin,
    WitchPlugin,
    HunterPlugin,
    VillagerPlugin
)

class TestWerewolfPlugin:
    def test_get_role_name(self):
        player = Player(player_id=0, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        plugin = WerewolfPlugin(player)
        assert plugin.get_role_name() == "狼人"
    
    def test_on_night_start(self):
        player = Player(player_id=0, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        plugin = WerewolfPlugin(player)
        
        players = [
            Player(player_id=0, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL),
            Player(player_id=1, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD),
            Player(player_id=2, name="村民2", role=RoleType.VILLAGER, team=Team.GOOD)
        ]
        state = GameState(game_id="test", phase=GamePhase.NIGHT_WOLF, day_number=1, players=players)
        
        action = plugin.on_night_start(state)
        assert action is not None
        assert action.type.value == "skill"
        assert action.target is not None
        assert action.target != 0  # 不能刀自己
    
    def test_get_valid_actions(self):
        player = Player(player_id=0, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        plugin = WerewolfPlugin(player)
        
        assert "刀人" in plugin.get_valid_actions(GamePhase.NIGHT_WOLF)
        assert "发言" in plugin.get_valid_actions(GamePhase.SPEECH)
        assert "投票" in plugin.get_valid_actions(GamePhase.VOTE)

class TestSeerPlugin:
    def test_get_role_name(self):
        player = Player(player_id=0, name="预言家", role=RoleType.SEER, team=Team.GOOD)
        plugin = SeerPlugin(player)
        assert plugin.get_role_name() == "预言家"
    
    def test_on_night_start(self):
        player = Player(player_id=0, name="预言家", role=RoleType.SEER, team=Team.GOOD)
        plugin = SeerPlugin(player)
        
        players = [
            Player(player_id=0, name="预言家", role=RoleType.SEER, team=Team.GOOD),
            Player(player_id=1, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL),
            Player(player_id=2, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD)
        ]
        state = GameState(game_id="test", phase=GamePhase.NIGHT_SEER, day_number=1, players=players)
        
        action = plugin.on_night_start(state)
        assert action is not None
        assert action.type.value == "skill"
        assert action.target is not None
        assert "查验结果" in action.content
    
    def test_win_condition(self):
        player = Player(player_id=0, name="预言家", role=RoleType.SEER, team=Team.GOOD)
        plugin = SeerPlugin(player)
        
        alive_players = [
            Player(player_id=0, name="预言家", role=RoleType.SEER, team=Team.GOOD, is_alive=True),
            Player(player_id=1, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD, is_alive=True)
        ]
        state = GameState(game_id="test", phase=GamePhase.DAY_END, day_number=1, players=alive_players)
        
        assert plugin.win_condition(state) is True

class TestWitchPlugin:
    def test_get_role_name(self):
        player = Player(player_id=0, name="女巫", role=RoleType.WITCH, team=Team.GOOD)
        plugin = WitchPlugin(player)
        assert plugin.get_role_name() == "女巫"
    
    def test_has_potions(self):
        player = Player(player_id=0, name="女巫", role=RoleType.WITCH, team=Team.GOOD)
        plugin = WitchPlugin(player)
        
        assert plugin.has_potion is True
        assert plugin.has_antidote is True
    
    def test_get_valid_actions(self):
        player = Player(player_id=0, name="女巫", role=RoleType.WITCH, team=Team.GOOD)
        plugin = WitchPlugin(player)
        
        actions = plugin.get_valid_actions(GamePhase.NIGHT_WITCH)
        assert "使用解药" in actions
        assert "使用毒药" in actions
        assert "不使用药水" in actions
    
    def test_use_potion(self):
        player = Player(player_id=0, name="女巫", role=RoleType.WITCH, team=Team.GOOD)
        plugin = WitchPlugin(player)
        
        players = [
            Player(player_id=0, name="女巫", role=RoleType.WITCH, team=Team.GOOD),
            Player(player_id=1, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        ]
        state = GameState(game_id="test", phase=GamePhase.NIGHT_WITCH, day_number=1, players=players)
        state.step_data = {"final_target": 1}
        
        action = plugin.on_night_start(state)
        assert action is not None
        
        plugin.has_antidote = False
        actions = plugin.get_valid_actions(GamePhase.NIGHT_WITCH)
        assert "使用解药" not in actions

class TestHunterPlugin:
    def test_get_role_name(self):
        player = Player(player_id=0, name="猎人", role=RoleType.HUNTER, team=Team.GOOD)
        plugin = HunterPlugin(player)
        assert plugin.get_role_name() == "猎人"
    
    def test_on_death(self):
        player = Player(player_id=0, name="猎人", role=RoleType.HUNTER, team=Team.GOOD)
        plugin = HunterPlugin(player)
        
        players = [
            Player(player_id=0, name="猎人", role=RoleType.HUNTER, team=Team.GOOD),
            Player(player_id=1, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        ]
        state = GameState(game_id="test", phase=GamePhase.DAY_END, day_number=1, players=players)
        
        action = plugin.on_death(state, is_poisoned=False)
        assert action is not None
        assert action.type.value == "skill"
        
        action = plugin.on_death(state, is_poisoned=True)
        assert action is None
    
    def test_cannot_shoot_twice(self):
        player = Player(player_id=0, name="猎人", role=RoleType.HUNTER, team=Team.GOOD)
        plugin = HunterPlugin(player)
        
        players = [
            Player(player_id=0, name="猎人", role=RoleType.HUNTER, team=Team.GOOD),
            Player(player_id=1, name="狼人1", role=RoleType.WEREWOLF, team=Team.EVIL)
        ]
        state = GameState(game_id="test", phase=GamePhase.DAY_END, day_number=1, players=players)
        
        plugin.on_death(state, is_poisoned=False)
        assert plugin.can_shoot is False
        
        action = plugin.on_death(state, is_poisoned=False)
        assert action is None

class TestVillagerPlugin:
    def test_get_role_name(self):
        player = Player(player_id=0, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD)
        plugin = VillagerPlugin(player)
        assert plugin.get_role_name() == "村民"
    
    def test_on_night_start(self):
        player = Player(player_id=0, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD)
        plugin = VillagerPlugin(player)
        
        state = GameState(game_id="test", phase=GamePhase.NIGHT_WOLF, day_number=1, players=[])
        
        action = plugin.on_night_start(state)
        assert action is None
    
    def test_get_valid_actions(self):
        player = Player(player_id=0, name="村民1", role=RoleType.VILLAGER, team=Team.GOOD)
        plugin = VillagerPlugin(player)
        
        assert "发言" in plugin.get_valid_actions(GamePhase.SPEECH)
        assert "投票" in plugin.get_valid_actions(GamePhase.VOTE)
        assert plugin.get_valid_actions(GamePhase.NIGHT_WOLF) == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
