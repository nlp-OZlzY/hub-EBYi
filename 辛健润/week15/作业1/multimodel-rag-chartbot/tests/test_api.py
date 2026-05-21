"""API Tests - Basic validation without external dependencies."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app

client = TestClient(app)


def test_health():
    """健康检查"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_files():
    """列出文件"""
    resp = client.get("/files")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_search_missing_q():
    """缺少查询参数应返回422"""
    resp = client.get("/search")
    assert resp.status_code == 422


def test_upload_missing_file():
    """缺少文件应返回422"""
    resp = client.post("/files/upload")
    assert resp.status_code == 422


def test_file_not_found():
    """文件不存在返回404"""
    resp = client.get("/files/99999")
    assert resp.status_code == 404


def test_invalid_file_id_type():
    """无效的文件ID类型应返回422"""
    resp = client.get("/files/abc")
    assert resp.status_code == 422  # FastAPI参数校验失败


if __name__ == "__main__":
    pytest.main([__file__, "-v"])