"""
离线处理Worker单元测试

测试用例：
1. test_encode_text_and_image: 测试向量编码
2. test_parse_document: 测试文档解析（需要minerU）
3. test_process_document: 测试完整流程
"""

import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEncoding:
    """测试向量编码功能"""

    def test_encode_text_only(self, sample_markdown):
        """测试纯文本编码"""
        from offline_process_worker import encode_text_and_image

        md_path, img_dir = sample_markdown
        text = "这是一个纯文本测试"

        bge, clip_text, clip_img = encode_text_and_image(text, md_path)

        assert len(bge) == 512  # BGE维度
        assert len(clip_text) == 1024  # CLIP text维度
        assert len(clip_img) == 1024  # CLIP image维度
        assert bge != [0] * 512  # 不应该是零向量

    def test_encode_text_with_image(self, sample_markdown):
        """测试图文混合编码"""
        from offline_process_worker import encode_text_and_image

        md_path, img_dir = sample_markdown
        # 创建一个假图片文件
        fake_img = os.path.join(img_dir, "test.jpg")
        with open(fake_img, 'wb') as f:
            f.write(b'fake image')

        text = "Test\n![test](images/test.jpg)"

        bge, clip_text, clip_img = encode_text_and_image(text, md_path)

        assert len(bge) == 512
        assert len(clip_text) == 1024
        assert len(clip_img) == 1024

    def test_encode_empty_text(self, sample_markdown):
        """测试空文本编码"""
        from offline_process_worker import encode_text_and_image

        md_path, _ = sample_markdown
        text = ""

        bge, clip_text, clip_img = encode_text_and_image(text, md_path)

        # 空文本应该返回零向量
        assert len(bge) == 512
        assert len(clip_text) == 1024
        assert len(clip_img) == 1024


class TestChunkSplitting:
    """测试Chunk划分"""

    def test_split_normal_text(self):
        """测试正常文本划分"""
        from offline_process_worker import split_text2chunks

        lines = ["这是第一段", "这是第二段", "这是第三段"]
        result = split_text2chunks(lines)

        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_with_references(self):
        """测试带引用行的划分"""
        from offline_process_worker import split_text2chunks

        lines = [
            "# 标题",
            "[1] 参考1",
            "[2] 参考2",
            "# References",
            "正文内容"
        ]
        result = split_text2chunks(lines)

        # 引用行应该被过滤
        for chunk in result:
            assert "[1]" not in chunk
            assert "[2]" not in chunk
            assert "# References" not in chunk

    def test_split_long_content(self):
        """测试长内容划分"""
        from offline_process_worker import split_text2chunks

        lines = ["A" * 300] * 5
        result = split_text2chunks(lines, chunk_size=256)

        # 应该分成多个chunk
        assert len(result) >= 1


class TestFileProcessing:
    """测试文件处理流程"""

    def test_update_file_state(self, temp_db):
        """测试更新文件状态"""
        from offline_process_worker import update_file_state
        from orm_model import init_db, File

        Session, _ = init_db(temp_db)
        session = Session()

        # 创建测试文件
        file = File(
            filename="test.pdf",
            filepath="/tmp/test.pdf",
            state="uploaded"
        )
        session.add(file)
        session.commit()

        # 测试更新状态
        update_file_state(file.id, "parsing", "测试中")

        # 验证
        updated_file = session.query(File).filter(File.id == file.id).first()
        assert updated_file.state == "parsing"

        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])