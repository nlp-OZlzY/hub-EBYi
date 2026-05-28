"""测试公共Fixtures"""
import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

from main import app
from orm_model import Base


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """每个测试用例独立的临时目录"""
    os.chdir(tmp_path)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    yield


@pytest.fixture
def db_session():
    """内存SQLite，每个测试隔离"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    """FastAPI测试客户端"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_kafka():
    """Mock KafkaProducer"""
    with patch("services.file_service.KafkaProducer") as mock:
        producer_instance = Mock()
        mock.return_value = producer_instance
        yield producer_instance


@pytest.fixture
def mock_milvus():
    """Mock MilvusClient"""
    with patch("services.search_service.MilvusClient") as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_bge_model():
    """Mock bge模型，返回固定向量"""
    with patch("services.encode_service.EncodeService.encode_text") as mock:
        mock.return_value = [0.1] * 512
        yield mock


@pytest.fixture
def mock_clip_model():
    """Mock clip模型，返回固定向量"""
    with patch("services.encode_service.EncodeService.encode_text_clip") as mock:
        mock.return_value = [0.2] * 1024
        yield mock


@pytest.fixture
def mock_qwen_client():
    """Mock Qwen-VL客户端"""
    with patch("services.chat_service.ChatService.llm_client") as mock:
        mock.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="这是mock的回答"))]
        )
        yield mock


@pytest.fixture
def sample_pdf():
    """测试用PDF文件"""
    return ("test.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")


@pytest.fixture
def sample_docx():
    """测试用DOCX文件"""
    return ("test.docx", b"PK fake docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@pytest.fixture
def sample_txt():
    """测试用TXT文件"""
    return ("test.txt", b"Hello World\nThis is test content.", "text/plain")
