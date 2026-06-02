import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.summary_agent import SummaryAgent


def test_summary_agent_init_uses_llm_client():
    """测试 SummaryAgent 使用 LLMClient 初始化"""
    with patch("agent.summary_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.from_config.return_value = mock_instance

        agent = SummaryAgent()

        assert agent.client is mock_instance
        assert "复盘分析师" in agent.system_prompt


@pytest.mark.asyncio
async def test_generate_summary_uses_llm_client():
    """测试 generate_summary 调用 LLMClient"""
    with patch("agent.summary_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(
            return_value=(
                '{"summary": "本局表现不错", "strategies": "悍跳", '
                '"mistakes": "发言不够自信", "lessons": "要更坚定"}'
            )
        )
        MockClient.from_config.return_value = mock_instance

        agent = SummaryAgent()
        result = await agent.generate_summary(
            player_name="玩家1",
            role_name="狼人",
            camp="evil",
            winner="evil",
            personal_history="你首夜击杀了预言家\n你白天伪装发言",
        )

        assert result["summary"] == "本局表现不错"
        assert result["strategies"] == "悍跳"
        assert result["mistakes"] == "发言不够自信"
        assert result["lessons"] == "要更坚定"
        mock_instance.chat.assert_called_once()

        call_kwargs = mock_instance.chat.call_args.kwargs
        assert "复盘分析师" in call_kwargs["system"]
        assert "玩家1" in call_kwargs["user"]
        assert "狼人" in call_kwargs["user"]
        assert "你首夜击杀了预言家" in call_kwargs["user"]


@pytest.mark.asyncio
async def test_generate_summary_llm_failure_returns_fallback():
    """测试 LLM 调用失败时的降级处理"""
    with patch("agent.summary_agent.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(side_effect=Exception("API Error"))
        MockClient.from_config.return_value = mock_instance

        agent = SummaryAgent()
        result = await agent.generate_summary(
            player_name="玩家3",
            role_name="村民",
            camp="good",
            winner="good",
            personal_history="你白天正常发言投票",
        )

        assert "总结生成失败" in result["summary"]
        assert result["strategies"] == ""


def test_parse_json_output_falls_back_to_summary_text():
    """测试非 JSON 输出降级为 summary 文本"""
    with patch("agent.summary_agent.LLMClient") as MockClient:
        MockClient.from_config.return_value = MagicMock()
        agent = SummaryAgent()

    result = agent._parse_json_output("纯文本回复，不是JSON格式")

    assert "纯文本回复" in result["summary"]
    assert result["strategies"] == ""
