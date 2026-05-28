"""Integration tests — end-to-end flow with mocked external services."""

import numpy as np
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api_server import app
    return TestClient(app)


def test_upload_then_chat_flow(client):
    """Upload a file, then chat — verify the full pipeline works with mocks."""
    # Step 1: Upload
    pdf_content = b"%PDF-1.4 fake content"
    with patch("api_server.Session") as MockSession, \
         patch("api_server.get_kafka_producer") as MockProducer:
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.id = 10
        mock_session.add.side_effect = lambda r: setattr(r, "id", 10)
        MockSession.return_value.__enter__ = lambda s: mock_session
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        MockProducer.return_value = MagicMock()

        upload_resp = client.post(
            "/upload/document",
            files={"file": ("report.pdf", pdf_content, "application/pdf")},
        )

    assert upload_resp.status_code == 200
    assert upload_resp.json()["status"] == "success"

    # Step 2: Chat
    mock_bge = MagicMock()
    mock_bge.encode.return_value = np.random.randn(512).astype(np.float32)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [[
        {"entity": {
            "text": "Report content about AI",
            "db_id": 10,
            "file_name": "report.pdf",
            "file_path": "/uploads/report.pdf",
        }}
    ]]

    mock_qwen = MagicMock()
    mock_qwen.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The report is about AI."))]
    )

    with patch("api_server.get_bge_model", return_value=mock_bge), \
         patch("api_server.get_milvus_client", return_value=mock_milvus), \
         patch("openai.OpenAI", return_value=mock_qwen):
        chat_resp = client.post("/chat", json={"question": "What is the report about?"})

    assert chat_resp.status_code == 200
    assert "AI" in chat_resp.json()["answer"]
    assert chat_resp.json()["sources"][0]["db_id"] == 10


def test_delete_removes_vectors(client):
    """Delete a file and verify Milvus delete is called."""
    mock_record = MagicMock()
    mock_record.filepath = "/tmp/fake.pdf"

    with patch("api_server.Session") as MockSession, \
         patch("api_server.get_milvus_client") as MockMilvus, \
         patch("os.path.exists", return_value=False), \
         patch("os.remove"):
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_record
        MockSession.return_value.__enter__ = lambda s: mock_session
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        mock_milvus = MagicMock()
        MockMilvus.return_value = mock_milvus

        resp = client.delete("/files/10")

    assert resp.status_code == 200
    mock_milvus.delete.assert_called_once()
    call_kwargs = mock_milvus.delete.call_args
    assert "db_id == 10" in str(call_kwargs)
