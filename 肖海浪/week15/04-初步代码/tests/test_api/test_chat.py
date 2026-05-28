"""问答接口测试"""
import pytest
from unittest.mock import patch, Mock


class TestChat:
    """POST /chat 接口测试"""

    def test_chat_success(self, client, mock_bge_model, mock_milvus, mock_qwen_client):
        """正常问答"""
        mock_milvus.search.return_value = [[
            {
                "entity": {
                    "text": "RAG是检索增强生成的缩写",
                    "db_id": 1,
                    "file_name": "test.pdf",
                    "file_path": "uploads/abc.pdf"
                },
                "distance": 0.85
            }
        ]]

        response = client.post(
            "/chat",
            json={"question": "什么是RAG？"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "answer" in data["data"]
        assert "sources" in data["data"]

    def test_chat_empty_question(self, client):
        """空问题"""
        response = client.post("/chat", json={"question": ""})

        assert response.status_code == 422

    def test_chat_missing_question_field(self, client):
        """缺少question字段"""
        response = client.post("/chat", json={})

        assert response.status_code == 422

    def test_chat_no_search_results(self, client, mock_bge_model, mock_milvus, mock_qwen_client):
        """Milvus无匹配结果"""
        mock_milvus.search.return_value = [[]]

        response = client.post(
            "/chat",
            json={"question": "完全无关的内容xyz123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["sources"] == []
