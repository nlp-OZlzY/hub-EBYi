"""ORM模型单元测试"""
import pytest
from orm_model import File


class TestFileORM:
    """File ORM模型测试"""

    def test_create_file_record(self, db_session):
        """创建文件记录"""
        file = File(
            filename="test.pdf",
            filepath="uploads/abc123.pdf",
            filestate="已上传"
        )
        db_session.add(file)
        db_session.commit()

        assert file.id is not None
        assert file.filename == "test.pdf"
        assert file.filepath == "uploads/abc123.pdf"
        assert file.filestate == "已上传"

    def test_query_file(self, db_session):
        """查询文件记录"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()

        result = db_session.query(File).filter(File.id == file.id).first()
        assert result is not None
        assert result.filename == "test.pdf"

    def test_update_file_state(self, db_session):
        """更新文件状态"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()

        file.filestate = "已解析"
        db_session.commit()

        result = db_session.query(File).filter(File.id == file.id).first()
        assert result.filestate == "已解析"

    def test_delete_file_record(self, db_session):
        """删除文件记录"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()
        file_id = file.id

        db_session.delete(file)
        db_session.commit()

        result = db_session.query(File).filter(File.id == file_id).first()
        assert result is None

    def test_query_all_files(self, db_session):
        """查询所有文件"""
        for i in range(3):
            file = File(
                filename=f"test{i}.pdf",
                filepath=f"uploads/abc{i}.pdf",
                filestate="已上传"
            )
            db_session.add(file)
        db_session.commit()

        results = db_session.query(File).all()
        assert len(results) == 3

    def test_file_state_transitions(self, db_session):
        """文件状态流转：已上传 -> 解析中 -> 已解析"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()

        file.filestate = "解析中"
        db_session.commit()
        assert file.filestate == "解析中"

        file.filestate = "已解析"
        db_session.commit()
        assert file.filestate == "已解析"

    def test_file_state_failed(self, db_session):
        """文件解析失败状态"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()

        file.filestate = "解析失败"
        db_session.commit()
        assert file.filestate == "解析失败"

    def test_to_dict(self, db_session):
        """测试to_dict方法"""
        file = File(filename="test.pdf", filepath="uploads/abc.pdf", filestate="已上传")
        db_session.add(file)
        db_session.commit()

        result = file.to_dict()
        assert result["id"] == file.id
        assert result["filename"] == "test.pdf"
        assert result["filepath"] == "uploads/abc.pdf"
        assert result["filestate"] == "已上传"
