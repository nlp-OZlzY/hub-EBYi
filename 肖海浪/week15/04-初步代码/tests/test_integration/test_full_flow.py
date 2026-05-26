"""集成测试"""
import pytest
import concurrent.futures
from unittest.mock import patch, Mock


class TestFullFlow:
    """完整业务流程集成测试"""

    def test_upload_to_search_flow(self, client, mock_kafka, mock_milvus,
                                    mock_bge_model, mock_qwen_client):
        """上传 -> 检索 完整流程"""
        # 1. 上传文件
        upload_resp = client.post(
            "/upload/document",
            files={"file": ("test.pdf", b"pdf content", "application/pdf")}
        )
        assert upload_resp.status_code == 200
        file_id = upload_resp.json()["data"]["file_id"]
        assert upload_resp.json()["data"]["filestate"] == "已上传"

        # 2. 验证文件列表
        list_resp = client.get("/files")
        assert len(list_resp.json()["data"]) == 1
        assert list_resp.json()["data"][0]["id"] == file_id

        # 3. Mock Milvus有数据
        mock_milvus.search.return_value = [[
            {
                "entity": {
                    "text": "测试文档内容",
                    "db_id": file_id,
                    "file_name": "test.pdf",
                    "file_path": "uploads/abc.pdf"
                },
                "distance": 0.9
            }
        ]]

        # 4. 提问
        chat_resp = client.post("/chat", json={"question": "测试文档说了什么？"})
        assert chat_resp.status_code == 200
        assert "answer" in chat_resp.json()["data"]

    def test_upload_delete_search_flow(self, client, mock_kafka, mock_milvus,
                                        mock_bge_model, mock_qwen_client):
        """上传 -> 删除 -> 检索 流程"""
        # 1. 上传
        upload_resp = client.post(
            "/upload/document",
            files={"file": ("test.pdf", b"pdf content", "application/pdf")}
        )
        file_id = upload_resp.json()["data"]["file_id"]

        # 2. 删除
        delete_resp = client.delete(f"/files/{file_id}")
        assert delete_resp.status_code == 200

        # 3. 验证文件列表为空
        list_resp = client.get("/files")
        assert len(list_resp.json()["data"]) == 0

        # 4. 提问，验证搜不到已删除文件
        mock_milvus.search.return_value = [[]]

        chat_resp = client.post("/chat", json={"question": "测试文档内容"})
        assert chat_resp.status_code == 200
        assert chat_resp.json()["data"]["sources"] == []

    def test_multi_file_search(self, client, mock_kafka, mock_milvus,
                                 mock_bge_model, mock_qwen_client):
        """多文件上传 -> 跨文件检索"""
        # 1. 上传3个文件
        file_ids = []
        for i in range(3):
            resp = client.post(
                "/upload/document",
                files={"file": (f"doc{i}.pdf", b"content", "application/pdf")}
            )
            file_ids.append(resp.json()["data"]["file_id"])

        # 2. Mock Milvus返回多个文件的结果
        mock_milvus.search.return_value = [[
            {
                "entity": {
                    "text": f"文档{i}的内容",
                    "db_id": file_ids[i],
                    "file_name": f"doc{i}.pdf",
                    "file_path": f"uploads/file{i}.pdf"
                },
                "distance": 0.9 - i * 0.1
            }
            for i in range(3)
        ]]

        # 3. 提问
        chat_resp = client.post("/chat", json={"question": "所有文档的汇总"})

        assert chat_resp.status_code == 200
        sources = chat_resp.json()["data"]["sources"]
        source_files = [s["file_name"] for s in sources]
        assert len(set(source_files)) > 1

    def test_concurrent_uploads(self, client, mock_kafka):
        """并发上传测试"""
        def upload_file(index):
            return client.post(
                "/upload/document",
                files={"file": (f"test{index}.pdf", b"content", "application/pdf")}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_file, i) for i in range(10)]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count == 10

        list_resp = client.get("/files")
        assert len(list_resp.json()["data"]) == 10
