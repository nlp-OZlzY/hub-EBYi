"""
测试数据管理接口
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_upload_text():
    """测试文本上传"""
    response = client.post(
        "/api/v1/data/upload",
        data={"content": "这是一段测试文本"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["msg"] == "上传成功"


def test_upload_file():
    """测试文件上传"""
    response = client.post(
        "/api/v1/data/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200


def test_delete_data():
    """测试数据删除"""
    # 先上传获取ID
    upload_resp = client.post("/api/v1/data/upload", data={"content": "to_delete"})
    item_id = upload_resp.json()["data"]["id"]

    # 删除
    response = client.post("/api/v1/data/delete", json=[item_id])
    assert response.status_code == 200
    assert response.json()["code"] == 200


def test_list_data():
    """测试数据列表查询"""
    response = client.get("/api/v1/data/list?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "items" in data["data"]