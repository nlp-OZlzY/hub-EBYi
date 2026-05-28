"""Tests for POST /chat endpoint."""

import numpy as np
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api_server import app
    return TestClient(app)


def _mock_bge():
    """Mock BGE model that returns a numpy array (so .tolist() works)."""
    mock = MagicMock()
    mock.encode.return_value = np.random.randn(512).astype(np.float32)
    return mock


def test_chat_success(client):
    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [[
        {"entity": {"text": "chunk text", "db_id": 1, "file_name": "test.pdf", "file_path": "/uploads/test.pdf"}}
    ]]

    mock_qwen = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "This is the answer."
    mock_qwen.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("api_server.get_bge_model", return_value=_mock_bge()), \
         patch("api_server.get_milvus_client", return_value=mock_milvus), \
         patch("openai.OpenAI", return_value=mock_qwen):
        resp = client.post("/chat", json={"question": "What is this about?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "This is the answer."
    assert len(data["sources"]) > 0
    assert "file_name" in data["sources"][0]


def test_chat_empty_question_returns_422(client):
    """Pydantic min_length=1 validation rejects empty string before endpoint runs."""
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422


def test_chat_sources_contain_file_info(client):
    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [[
        {"entity": {"text": "some content", "db_id": 5, "file_name": "doc.pdf", "file_path": "/uploads/doc.pdf"}}
    ]]

    mock_qwen = MagicMock()
    mock_qwen.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="answer"))]
    )

    with patch("api_server.get_bge_model", return_value=_mock_bge()), \
         patch("api_server.get_milvus_client", return_value=mock_milvus), \
         patch("openai.OpenAI", return_value=mock_qwen):
        resp = client.post("/chat", json={"question": "test"})

    sources = resp.json()["sources"]
    assert sources[0]["db_id"] == 5
    assert sources[0]["file_name"] == "doc.pdf"


def test_chat_top_k_parameter(client):
    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [[]]

    mock_qwen = MagicMock()
    mock_qwen.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="no results"))]
    )

    with patch("api_server.get_bge_model", return_value=_mock_bge()), \
         patch("api_server.get_milvus_client", return_value=mock_milvus), \
         patch("openai.OpenAI", return_value=mock_qwen):
        resp = client.post("/chat", json={"question": "test", "top_k": 3})

    assert resp.status_code == 200
    mock_milvus.search.assert_called_once()
    call_kwargs = mock_milvus.search.call_args
    assert call_kwargs[1]["limit"] == 3 or call_kwargs.kwargs.get("limit") == 3
