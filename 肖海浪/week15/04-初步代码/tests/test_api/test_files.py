"""文件管理接口测试"""
import pytest
from unittest.mock import patch


class TestListFiles:
    """GET /files 接口测试"""

    def test_list_files_empty(self, client):
        """无文件时返回空列表"""
        response = client.get("/files")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"] == []

    def test_list_files_with_data(self, client, sample_pdf, mock_kafka):
        """有文件时返回列表"""
        client.post("/upload/document", files={"file": sample_pdf})

        response = client.get("/files")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["filename"] == "test.pdf"

    def test_list_files_fields(self, client, sample_pdf, mock_kafka):
        """验证返回字段完整性"""
        client.post("/upload/document", files={"file": sample_pdf})

        response = client.get("/files")
        file_data = response.json()["data"][0]

        assert "id" in file_data
        assert "filename" in file_data
        assert "filepath" in file_data
        assert "filestate" in file_data


class TestDeleteFile:
    """DELETE /files/{file_id} 接口测试"""

    def test_delete_success(self, client, sample_pdf, mock_kafka, mock_milvus):
        """删除存在的文件"""
        upload_resp = client.post("/upload/document", files={"file": sample_pdf})
        file_id = upload_resp.json()["data"]["file_id"]

        response = client.delete(f"/files/{file_id}")

        assert response.status_code == 200

        list_resp = client.get("/files")
        assert len(list_resp.json()["data"]) == 0

    def test_delete_not_found(self, client):
        """删除不存在的文件"""
        response = client.delete("/files/99999")

        assert response.status_code == 404

    def test_delete_twice(self, client, sample_pdf, mock_kafka, mock_milvus):
        """重复删除同一文件"""
        upload_resp = client.post("/upload/document", files={"file": sample_pdf})
        file_id = upload_resp.json()["data"]["file_id"]

        response1 = client.delete(f"/files/{file_id}")
        assert response1.status_code == 200

        response2 = client.delete(f"/files/{file_id}")
        assert response2.status_code == 404
