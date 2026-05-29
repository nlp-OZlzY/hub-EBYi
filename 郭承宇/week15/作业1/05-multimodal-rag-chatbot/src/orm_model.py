"""SQLAlchemy ORM model for file metadata stored in SQLite."""

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DB_PATH

Base = declarative_base()


class FileRecord(Base):
    """File metadata table."""

    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    filestate = Column(String(20), nullable=False, default="已上传")

    __table_args__ = ({"sqlite_autoincrement": True},)


engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
