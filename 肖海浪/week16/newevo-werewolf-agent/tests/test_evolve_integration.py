import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from prompt_store.store import PromptStore

def test_prompt_store_version_growth(tmp_path):
    """测试 prompt 版本随演化增长"""
    prompts_dir = str(tmp_path / "prompts" / "roles")
    versions_dir = str(tmp_path / "prompt_versions")
    os.makedirs(prompts_dir, exist_ok=True)

    store = PromptStore(prompts_dir=prompts_dir, versions_dir=versions_dir)

    store.write_prompt("werewolf", "# 狼人策略 v1")

    for i in range(3):
        current = store.read_prompt("werewolf")
        store.save_version("werewolf", current)
        store.write_prompt("werewolf", f"# 狼人策略 v{i+2}")

    versions = store.list_versions("werewolf")
    assert len(versions) == 3

    current = store.read_prompt("werewolf")
    assert "v4" in current


def test_player_agent_loads_from_prompt_file(tmp_path):
    """测试 PlayerAgent 从 prompt 文件加载"""
    from agent.player_agent import PlayerAgent

    prompt_file = tmp_path / "werewolf.md"
    prompt_file.write_text("# 自定义狼人策略\n\n你是超级狼人。", encoding="utf-8")

    with patch("agent.player_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.from_config.return_value = mock_instance

        agent = PlayerAgent(
            player_id=0,
            role_name="狼人",
            private_context={"role": "Werewolf", "player_id": 0, "camp": "evil"},
            camp="evil",
            prompt_file=str(prompt_file),
        )
        instructions = agent._build_instructions()
        assert "超级狼人" in instructions


def test_metrics_collector_with_mock_engine():
    """测试 MetricsCollector 与模拟引擎"""
    from metrics.collector import MetricsCollector

    mock_engine = MagicMock()
    mock_engine.game_state.get_winner.return_value = "good"
    mock_engine.game_state.day_number = 3
    mock_engine.game_state.dialogues = [
        {"action": "speech", "player_id": 0, "content": "我是好人", "day": 1},
        {"action": "vote", "player_id": 0, "target": 1, "day": 1},
    ]
    mock_engine.death_records = []

    mock_player = MagicMock()
    mock_player.player_id = 0
    mock_player.role.role_type.value = "werewolf"
    mock_player.role.camp.value = "evil"
    mock_player.to_dict.return_value = {"player_id": 0, "role": "werewolf", "camp": "evil"}
    mock_engine.game_state.players = [mock_player]

    results = MetricsCollector.collect(mock_engine)
    assert "werewolf" in results
    assert results["werewolf"]["metrics"]["win"] is False


@pytest.mark.asyncio
async def test_evolve_loop_updates_prompt(tmp_path):
    """测试 evolve_loop 在反思器返回新 prompt 时写入版本历史"""
    prompts_dir = str(tmp_path / "prompts" / "roles")
    versions_dir = str(tmp_path / "prompt_versions")
    os.makedirs(prompts_dir, exist_ok=True)
    store = PromptStore(prompts_dir=prompts_dir, versions_dir=versions_dir)
    store.write_prompt("werewolf", "# 狼人策略 v1")

    mock_player = MagicMock()
    mock_player.player_id = 0
    mock_player.role.role_type.value = "werewolf"
    mock_player.name = "玩家1"

    mock_engine = MagicMock()
    mock_engine.game_state.is_game_over.return_value = True
    mock_engine.game_state.get_winner.return_value = "evil"
    mock_engine.game_state.players = [mock_player]
    mock_engine.death_records = []
    mock_engine.game_state.get_player_private_context.return_value = {
        "dialogues": [{"action": "speech", "content": "测试发言", "day": 1}],
    }
    mock_engine.initialize = AsyncMock()

    mock_reflector = MagicMock()
    mock_reflector.reflect = AsyncMock(return_value="# 狼人策略 v2\n\n改进版")

    metrics_payload = {
        "werewolf": {
            "role": "werewolf",
            "player_id": 0,
            "metrics": {"win": True, "survival_days": 2},
            "highlights": [],
        }
    }

    with patch("evolve.PromptStore", return_value=store), \
         patch("evolve.SelfReflector", return_value=mock_reflector), \
         patch("evolve.GameEngine", return_value=mock_engine), \
         patch("evolve.MetricsCollector.collect", return_value=metrics_payload), \
         patch("evolve.shuffle_roles", side_effect=lambda x: x), \
         patch("evolve._read_agent_md", side_effect=lambda r: store.read_prompt(r)), \
         patch("evolve._write_agent_md", side_effect=lambda r, c: store.write_prompt(r, c)):
        from evolve import evolve_loop

        await evolve_loop(rounds=1, config_name="simple_4", shuffle=False)

    assert "v2" in store.read_prompt("werewolf")
    assert len(store.list_versions("werewolf")) == 1
    mock_reflector.reflect.assert_called_once()
