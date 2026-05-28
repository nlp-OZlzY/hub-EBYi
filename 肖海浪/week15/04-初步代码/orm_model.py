"""SQLAlchemy ORM 模型"""
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

Base = declarative_base()


class File(Base):
    """文件表"""
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)  # 原始文件名
    filepath = Column(String(1000), nullable=False)  # 存储路径
    filestate = Column(String(20), nullable=False)   # 状态：已上传/解析中/已解析/解析失败

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "filepath": self.filepath,
            "filestate": self.filestate
        }


# 数据库初始化
engine = create_engine(f'sqlite:///{settings.DB_PATH}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
