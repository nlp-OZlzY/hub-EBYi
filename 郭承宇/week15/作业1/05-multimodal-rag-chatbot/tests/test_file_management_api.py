"""Tests for GET /files and DELETE /files/{file_id} endpoints."""

import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api_server import app
    return TestClient(app)


def test_list_files(client):
    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.filename = "test.pdf"
    mock_file.filepath = "/uploads/test.pdf"
    mock_file.filestate = "已完成"

    with patch("api_server.Session") as MockSession:
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_file]
        MockSession.return_value.__enter__ = lambda s: mock_session
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/files")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["filename"] == "test.pdf"
    assert data[0]["filestate"] == "已完成"


def test_delete_file_success(client):
    mock_record = MagicMock()
    mock_record.filepath = "/tmp/nonexistent_file_for_test.pdf"

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

        resp = client.delete("/files/1")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_delete_nonexistent_file_returns_404(client):
    with patch("api_server.Session") as MockSession:
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        MockSession.return_value.__enter__ = lambda s: mock_session
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.delete("/files/99999")

    assert resp.status_code == 404
