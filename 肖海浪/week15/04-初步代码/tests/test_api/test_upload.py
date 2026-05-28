"""上传接口测试"""
import pytest
import os
from unittest.mock import patch, Mock


class TestUploadDocument:
    """POST /upload/document 接口测试"""

    def test_upload_pdf_success(self, client, sample_pdf, mock_kafka):
        """正常上传PDF文件"""
        response = client.post(
            "/upload/document",
            files={"file": sample_pdf}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["file_id"] is not None
        assert data["data"]["filename"] == "test.pdf"
        assert data["data"]["filestate"] == "已上传"

    def test_upload_docx_success(self, client, sample_docx, mock_kafka):
        """正常上传DOCX文件"""
        response = client.post(
            "/upload/document",
            files={"file": sample_docx}
        )

        assert response.status_code == 200
        assert response.json()["data"]["filename"] == "test.docx"

    def test_upload_txt_success(self, client, sample_txt, mock_kafka):
        """正常上传TXT文件"""
        response = client.post(
            "/upload/document",
            files={"file": sample_txt}
        )

        assert response.status_code == 200
        assert response.json()["data"]["filename"] == "test.txt"

    def test_upload_unsupported_type(self, client):
        """上传不支持的文件类型"""
        response = client.post(
            "/upload/document",
            files={"file": ("test.exe", b"content", "application/octet-stream")}
        )

        assert response.status_code == 400

    def test_upload_empty_file(self, client):
        """上传空文件"""
        response = client.post(
            "/upload/document",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )

        assert response.status_code == 400

    def test_upload_no_file_field(self, client):
        """请求中无file字段"""
        response = client.post("/upload/document")

        assert response.status_code == 422
