"""
测试多模态检索接口
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_retrieval_by_text():
    """测试文本检索"""
    response = client.post("/api/v1/retrieval/text?query=深度学习&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "results" in data["data"]
    assert len(data["data"]["results"]) <= 3


def test_retrieval_by_condition():
    """测试条件检索"""
    response = client.post(
        "/api/v1/retrieval/condition?query=Transformer&doc_type=pdf&limit=5"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["data"]["filters"]["doc_type"] == "pdf"