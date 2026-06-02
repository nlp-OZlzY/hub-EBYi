import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.player_agent import PlayerAgent, create_player_agent


def test_player_agent_init_with_prompt_file(tmp_path):
    """测试 PlayerAgent 从 prompt 文件加载"""
    prompt_file = tmp_path / "werewolf.md"
    prompt_file.write_text("# 狼人策略\n\n你是狼人。", encoding="utf-8")

    with patch("agent.player_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.from_config.return_value = mock_instance

        agent = PlayerAgent(
            player_id=0,
            role_name="狼人",
            private_context={"role": "Werewolf", "player_id": 0, "camp": "evil"},
            camp="evil",
            decision_style="balanced",
            role_type="werewolf",
            prompt_file=str(prompt_file),
        )
        assert "狼人策略" in agent._build_instructions()


def test_player_agent_init_without_prompt_file():
    """测试 PlayerAgent 没有 prompt 文件时使用默认行为"""
    with patch("agent.player_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.from_config.return_value = mock_instance

        agent = PlayerAgent(
            player_id=0,
            role_name="狼人",
            private_context={"role": "Werewolf", "player_id": 0, "camp": "evil"},
            camp="evil",
            decision_style="balanced",
            role_type="werewolf",
        )
        instructions = agent._build_instructions()
        assert "狼人" in instructions


@pytest.mark.asyncio
async def test_player_agent_decide_night_action():
    """测试 PlayerAgent 夜间决策使用 LLMClient"""
    with patch("agent.player_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(
            return_value='{"action": "night_action", "target": 3, "reasoning": "测试"}'
        )
        MockClient.from_config.return_value = mock_instance

        agent = PlayerAgent(
            player_id=0,
            role_name="狼人",
            private_context={"role": "Werewolf", "player_id": 0, "camp": "evil"},
            camp="evil",
        )
        result = await agent.decide_night_action({"alive_players": [0, 1, 2, 3]})
        assert result["action"] == "night_action"
        assert result["target"] == 3
        mock_instance.chat.assert_called_once()


def test_create_player_agent_passes_prompt_file(tmp_path):
    """测试工厂函数传递 prompt_file"""
    prompt_file = tmp_path / "seer.md"
    prompt_file.write_text("# 预言家策略\n\n你是预言家。", encoding="utf-8")

    with patch("agent.player_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.from_config.return_value = mock_instance

        agent = create_player_agent(
            player_id=1,
            role_name="预言家",
            private_context={"role": "Seer", "player_id": 1, "camp": "good"},
            camp="good",
            role_type="seer",
            prompt_file=str(prompt_file),
        )
        assert agent.prompt_file == str(prompt_file)
        assert "预言家策略" in agent._build_instructions()
