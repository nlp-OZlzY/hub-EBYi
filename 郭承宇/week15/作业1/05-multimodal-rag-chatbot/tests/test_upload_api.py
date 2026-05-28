"""Tests for POST /upload/document endpoint."""

import io
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api_server import app
    return TestClient(app)


def _mock_session():
    """Create a mock SQLAlchemy session context manager."""
    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.id = 42
    mock_session.query.return_value.filter.return_value.first.return_value = mock_record
    return mock_session


def test_upload_pdf_success(client):
    pdf_content = b"%PDF-1.4 fake pdf content"
    with patch("api_server.Session") as MockSession, \
         patch("api_server.get_kafka_producer") as MockProducer:
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.id = 1
        mock_session.add.side_effect = lambda r: setattr(r, "id", 1)
        MockSession.return_value.__enter__ = lambda s: mock_session
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        producer = MagicMock()
        MockProducer.return_value = producer

        resp = client.post(
            "/upload/document",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "file_id" in data


def test_upload_empty_file_returns_400(client):
    resp = client.post(
        "/upload/document",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


def test_upload_unsupported_type_returns_400(client):
    resp = client.post(
        "/upload/document",
        files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert resp.status_code == 400
