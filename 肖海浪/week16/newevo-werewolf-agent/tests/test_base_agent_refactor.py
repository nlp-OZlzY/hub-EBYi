import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent.base import BaseAgent


def test_base_agent_init():
    with patch("agent.base.LLMClient") as MockClient:
        agent = BaseAgent()
        assert agent.client is not None


@pytest.mark.asyncio
async def test_base_agent_run():
    with patch("agent.base.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(return_value="测试回复")
        MockClient.from_config.return_value = mock_instance

        agent = BaseAgent()
        result = await agent.run("测试输入")
        assert result == "测试回复"
        mock_instance.chat.assert_called_once()
