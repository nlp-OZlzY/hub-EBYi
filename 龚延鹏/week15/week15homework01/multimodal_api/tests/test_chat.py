"""
测试多模态问答接口
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_basic_chat():
    """测试基础问答"""
    response = client.post("/api/v1/chat/basic?query=什么是深度学习")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "answer" in data["data"]


def test_basic_chat_with_keywords():
    """测试带关键词的基础问答"""
    response = client.post("/api/v1/chat/basic?query=请介绍一下RAG技术")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "RAG" in data["data"]["answer"]


def test_chat_with_retrieval():
    """测试带检索的问答"""
    response = client.post("/api/v1/chat/with_retrieval", json={
        "query": "CLIP模型的作用是什么？",
        "retrieval_context": [
            {"id": 1, "text": "CLIP是图文跨模态模型", "doc_name": "多模态模型.pdf", "page": 5}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["data"]["context_count"] == 1
    assert "CLIP" in data["data"]["answer"]