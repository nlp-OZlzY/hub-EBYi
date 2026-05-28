"""Tests for orm_model.py — File model CRUD with temp SQLite."""

import os
import sys
import tempfile

import pytest

# Import config first, patch DB_PATH before orm_model loads
import config

@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary SQLite database for each test."""
    old_path = config.DB_PATH
    db_file = str(tmp_path / "test.db")
    config.DB_PATH = db_file

    # Re-create engine and tables with temp DB
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import orm_model
    test_engine = create_engine(f"sqlite:///{db_file}", echo=False)
    orm_model.Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine)

    old_session_factory = orm_model.Session
    old_engine = orm_model.engine
    orm_model.engine = test_engine
    orm_model.Session = TestSession

    yield TestSession

    orm_model.DB_PATH = old_path
    config.DB_PATH = old_path
    orm_model.Session = old_session_factory
    orm_model.engine = old_engine


def test_create_file_record(temp_db):
    from orm_model import FileRecord as File
    session = temp_db()
    record = File(filename="test.pdf", filepath="/uploads/test.pdf", filestate="已上传")
    session.add(record)
    session.commit()

    assert record.id is not None
    assert record.filename == "test.pdf"
    assert record.filestate == "已上传"
    session.close()


def test_query_file_record(temp_db):
    from orm_model import FileRecord as File
    session = temp_db()
    record = File(filename="doc.pdf", filepath="/uploads/doc.pdf", filestate="已上传")
    session.add(record)
    session.commit()

    result = session.query(File).filter(File.id == record.id).first()
    assert result is not None
    assert result.filename == "doc.pdf"
    session.close()


def test_update_filestate(temp_db):
    from orm_model import FileRecord as File
    session = temp_db()
    record = File(filename="a.pdf", filepath="/a", filestate="已上传")
    session.add(record)
    session.commit()

    record.filestate = "解析中"
    session.commit()

    updated = session.query(File).filter(File.id == record.id).first()
    assert updated.filestate == "解析中"
    session.close()


def test_delete_file_record(temp_db):
    from orm_model import FileRecord as File
    session = temp_db()
    record = File(filename="b.pdf", filepath="/b", filestate="已上传")
    session.add(record)
    session.commit()
    rid = record.id

    session.delete(record)
    session.commit()

    assert session.query(File).filter(File.id == rid).first() is None
    session.close()
