import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent.reflector import SelfReflector, _load_reflector_meta_prompt


def test_reflector_loads_meta_prompt_from_file():
    """测试优先从 prompts/meta/reflector.md 加载元提示词"""
    meta = _load_reflector_meta_prompt()
    assert "反思器" in meta
    assert "自演化" in meta or "改进" in meta


@pytest.mark.asyncio
async def test_reflector_reads_current_prompt():
    """测试反思器能正确接收当前 prompt"""
    with patch("agent.reflector.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(return_value="# 改进后的狼人策略\n\n你是狼人。改进版。")
        MockClient.from_config.return_value = mock_instance

        reflector = SelfReflector()
        result = await reflector.reflect(
            role="werewolf",
            current_prompt="# 狼人策略\n\n你是狼人。",
            metrics={"win": False, "survival_days": 1},
            game_logs="第1天被投票出局",
        )
        assert "改进后" in result or "狼人" in result
        call_args = mock_instance.chat.call_args
        assert "你是狼人" in call_args[1]["user"] or "你是狼人" in str(call_args)


@pytest.mark.asyncio
async def test_reflector_output_is_valid_prompt():
    """测试反思器输出是完整的 prompt"""
    with patch("agent.reflector.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(
            return_value="# 狼人策略指令\n\n## 身份信息\n你是狼人。\n\n## 夜间策略\n- 改进的策略"
        )
        MockClient.from_config.return_value = mock_instance

        reflector = SelfReflector()
        result = await reflector.reflect(
            role="werewolf",
            current_prompt="# 狼人策略",
            metrics={"win": False},
            game_logs="日志",
        )
        assert len(result) > 10
        assert "狼人" in result


@pytest.mark.asyncio
async def test_reflector_preserves_structure():
    """测试反思器保留 prompt 结构"""
    structured_prompt = """# 狼人策略

## 身份信息
你是狼人。

## 夜间策略
- 策略1

## 白天策略
- 策略2"""

    with patch("agent.reflector.LLMClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock(return_value=structured_prompt)
        MockClient.from_config.return_value = mock_instance

        reflector = SelfReflector()
        result = await reflector.reflect(
            role="werewolf",
            current_prompt=structured_prompt,
            metrics={"win": True},
            game_logs="日志",
        )
        assert "## 身份信息" in result
        assert "## 夜间策略" in result
        assert "## 白天策略" in result
