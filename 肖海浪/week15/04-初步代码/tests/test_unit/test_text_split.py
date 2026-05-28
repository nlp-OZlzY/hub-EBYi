"""文本切分单元测试"""
import pytest
from utils.text_splitter import split_text2chunks


class TestSplitText2Chunks:
    """split_text2chunks 函数测试"""

    def test_empty_input(self):
        """空输入返回空列表"""
        result = split_text2chunks([])
        assert result == []

    def test_single_short_line(self):
        """单行短文本"""
        result = split_text2chunks(["这是一段测试文本"])
        assert len(result) == 1
        assert result[0] == "这是一段测试文本"

    def test_multiple_lines_within_limit(self):
        """多行文本，总长度未超过chunk_size"""
        lines = ["第一行", "第二行", "第三行"]
        result = split_text2chunks(lines, chunk_size=100)
        assert len(result) == 1
        assert "第一行" in result[0]
        assert "第三行" in result[0]

    def test_multiple_chunks_when_exceed_limit(self):
        """超过chunk_size时切分为多个chunk"""
        lines = ["a" * 200, "b" * 200, "c" * 200]
        result = split_text2chunks(lines, chunk_size=256)
        assert len(result) >= 2

    def test_skip_empty_lines(self):
        """跳过空行"""
        lines = ["第一行", "", "", "第四行"]
        result = split_text2chunks(lines)
        assert len(result) == 1
        assert result[0] == "第一行\n第四行"

    def test_skip_references_title(self):
        """跳过 # References 标题"""
        lines = ["正文内容", "# References", "参考文献1", "参考文献2"]
        result = split_text2chunks(lines)
        assert len(result) == 1
        assert "References" not in result[0]
        assert "参考文献" not in result[0]

    def test_skip_citation_lines(self):
        """跳过 [1] xxx 引用行"""
        lines = [
            "正文内容",
            "[1] Author et al. Title. 2024",
            "[2] Another Author. Title. 2023",
            "更多正文"
        ]
        result = split_text2chunks(lines)
        assert len(result) == 1
        assert "[1]" not in result[0]
        assert "[2]" not in result[0]
        assert "正文内容" in result[0]
        assert "更多正文" in result[0]

    def test_keep_image_lines(self):
        """保留图片引用行"""
        lines = ["文本内容", "![图1](./images/fig1.png)", "更多文本"]
        result = split_text2chunks(lines)
        assert len(result) == 1
        assert "![" in result[0]
        assert "fig1.png" in result[0]

    def test_chunk_size_boundary(self):
        """chunk_size边界测试"""
        lines = ["a" * 256, "b" * 10]
        result = split_text2chunks(lines, chunk_size=256)
        assert len(result) >= 1

    def test_real_world_document(self):
        """真实文档场景模拟"""
        lines = [
            "# 论文标题",
            "",
            "## 摘要",
            "本文提出了一种新的方法...",
            "",
            "## 1. 引言",
            "深度学习在NLP领域取得了巨大成功[1]。",
            "",
            "![图1](./images/fig1.png)",
            "",
            "# References",
            "[1] Vaswani et al. Attention is all you need. 2017",
            "[2] Devlin et al. BERT. 2019"
        ]
        result = split_text2chunks(lines, chunk_size=200)
        full_text = "\n".join(result)
        assert "论文标题" in full_text
        assert "摘要" in full_text
        assert "![图1]" in full_text
        assert "References" not in full_text
        assert "[1] Vaswani" not in full_text
