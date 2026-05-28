import pytest
import json
from llm.client import LLMClient, load_llm_config


def test_load_llm_config():
    config = load_llm_config("config/llm_config.json")
    assert "base_url" in config
    assert "model" in config
    assert isinstance(config["model"], str)
    assert len(config["model"]) > 0


def test_llm_client_init():
    client = LLMClient(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    assert client.model == "test-model"
    assert client.client is not None


def test_llm_client_from_config():
    client = LLMClient.from_config("config/llm_config.json", api_key="test-key")
    assert isinstance(client.model, str)
    assert len(client.model) > 0
