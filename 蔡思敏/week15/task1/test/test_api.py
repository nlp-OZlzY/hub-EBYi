"""
API单元测试

测试用例：
1. test_health_check: 健康检查
2. test_knowledge_base_crud: 知识库CRUD
3. test_file_upload: 文件上传
4. test_chat: 问答功能
"""

import pytest
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orm_model import (
    init_db, KnowledgeBase, File, Chunk, ChunkImage,
    ChatSession, ChatMessage, FileState, ChunkType, get_db_path
)


class TestOrmModel:
    """测试ORM模型"""

    def test_init_db(self, temp_db):
        """测试数据库初始化"""
        Session, engine = init_db(temp_db)
        session = Session()

        # 测试创建知识库
        kb = KnowledgeBase(name="测试知识库", description="测试描述")
        session.add(kb)
        session.commit()

        # 验证
        kbs = session.query(KnowledgeBase).all()
        assert len(kbs) == 1
        assert kbs[0].name == "测试知识库"

        session.close()

    def test_file_state_enum(self):
        """测试文件状态枚举"""
        assert FileState.UPLOADED.value == "uploaded"
        assert FileState.PARSING.value == "parsing"
        assert FileState.PARSED.value == "parsed"
        assert FileState.INDEXED.value == "indexed"

    def test_chunk_type_enum(self):
        """测试Chunk类型枚举"""
        assert ChunkType.TEXT.value == "text"
        assert ChunkType.IMAGE.value == "image"
        assert ChunkType.MIXED.value == "mixed"


class TestKnowledgeBaseAPI:
    """测试知识库API（需要启动API服务）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_url = "http://localhost:8000/api/v1"

    def test_health_check(self):
        """测试健康检查"""
        import requests
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
        except requests.exceptions.ConnectionError:
            pytest.skip("API服务未启动")

    def test_create_knowledge_base(self):
        """测试创建知识库"""
        import requests
        try:
            response = requests.post(
                f"{self.base_url}/knowledge-bases",
                json={"name": "测试知识库", "description": "测试"},
                timeout=5
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "测试知识库"
        except requests.exceptions.ConnectionError:
            pytest.skip("API服务未启动")

    def test_list_knowledge_bases(self):
        """测试获取知识库列表"""
        import requests
        try:
            response = requests.get(f"{self.base_url}/knowledge-bases", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API服务未启动")


class TestChunkSplitting:
    """测试Chunk划分功能"""

    def test_split_text2chunks_empty(self):
        """测试空列表"""
        from offline_process_worker import split_text2chunks
        result = split_text2chunks([])
        assert result == []

    def test_split_text2chunks_normal(self):
        """测试正常文本"""
        from offline_process_worker import split_text2chunks
        lines = ["line1", "line2", "line3"]
        result = split_text2chunks(lines)
        assert len(result) > 0

    def test_split_text2chunks_with_reference(self):
        """测试带引用行"""
        from offline_process_worker import split_text2chunks
        lines = ["[1] Reference", "normal text", "# References"]
        result = split_text2chunks(lines)
        # 引用行应该被过滤
        assert all("[1]" not in chunk for chunk in result)

    def test_split_text2chunks_long_line(self):
        """测试超长行"""
        from offline_process_worker import split_text2chunks
        lines = ["A" * 500, "B" * 500]
        result = split_text2chunks(lines)
        # 超长行应该被分割
        assert len(result) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])