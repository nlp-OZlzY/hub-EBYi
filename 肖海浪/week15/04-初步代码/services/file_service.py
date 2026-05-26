"""
文件管理服务

负责：
1. 保存上传的文件到本地
2. 创建数据库记录
3. 发送Kafka消息触发离线解析
4. 查询和删除文件
"""
import os
import uuid
import json
import logging
from typing import Optional
from kafka import KafkaProducer
from orm_model import File, Session
from config import settings

logger = logging.getLogger(__name__)


class FileService:
    """
    文件管理服务类

    职责：
    - 文件上传：保存文件 + 写数据库 + 发Kafka消息
    - 文件查询：从数据库查询文件列表
    - 文件删除：删除本地文件 + 删除数据库记录
    """

    def __init__(self):
        """
        初始化服务

        1. 创建上传目录
        2. 创建Kafka生产者
        """
        # 确保上传目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # 创建Kafka生产者
        # 用于发送消息到Kafka，触发Worker处理
        self.producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def save_and_publish(self, filename: str, content: bytes) -> dict:
        """
        保存文件并发送Kafka消息

        Args:
            filename: 原始文件名，例如 "report.pdf"
            content: 文件内容（二进制）

        Returns:
            文件信息字典，包含 file_id, filename, filestate

        流程：
        1. 生成唯一文件名（UUID避免冲突）
        2. 保存文件到本地
        3. 创建数据库记录
        4. 发送Kafka消息
        """
        # ============================================
        # 步骤1：生成唯一文件名
        # UUID格式：550e8400-e29b-41d4-a716-446655440000
        # 避免不同用户上传同名文件时冲突
        # ============================================
        ext = os.path.splitext(filename)[1]  # 获取文件扩展名，如 .pdf
        save_name = f"{uuid.uuid4()}{ext}"   # 生成唯一文件名
        save_path = os.path.join(settings.UPLOAD_DIR, save_name)  # 完整路径

        # ============================================
        # 步骤2：保存文件到本地
        # "wb" 表示以二进制写模式打开
        # ============================================
        with open(save_path, "wb") as f:
            f.write(content)
        logger.info(f"文件已保存: {save_path}")

        # ============================================
        # 步骤3：创建数据库记录
        # 使用 with Session() 自动管理数据库连接
        # ============================================
        with Session() as session:
            record = File(
                filename=filename,      # 原始文件名
                filepath=save_path,     # 存储路径
                filestate="已上传"      # 初始状态
            )
            session.add(record)    # 添加记录
            session.commit()       # 提交事务
            file_id = record.id    # 获取自增ID
        logger.info(f"数据库记录已创建: file_id={file_id}")

        # ============================================
        # 步骤4：发送Kafka消息
        # 消息会被Worker消费，触发文档解析
        # ============================================
        self.producer.send(settings.KAFKA_TOPIC, value={
            "file_name": filename,  # 文件名
            "file_path": save_path, # 文件路径
            "id": file_id           # 数据库ID
        })
        self.producer.flush()  # 确保消息发送完成
        logger.info(f"Kafka消息已发送: topic={settings.KAFKA_TOPIC}")

        return {
            "file_id": file_id,
            "filename": filename,
            "filestate": "已上传"
        }

    def get_all_files(self) -> list:
        """
        获取所有文件列表

        Returns:
            文件信息列表
        """
        with Session() as session:
            files = session.query(File).all()
            return [f.to_dict() for f in files]

    def get_file_by_id(self, file_id: int) -> Optional[dict]:
        """
        根据ID获取文件

        Args:
            file_id: 文件ID

        Returns:
            文件信息字典，不存在返回None
        """
        with Session() as session:
            file = session.query(File).filter(File.id == file_id).first()
            return file.to_dict() if file else None

    def delete_file(self, file_id: int) -> bool:
        """
        删除文件

        Args:
            file_id: 文件ID

        Returns:
            是否删除成功

        流程：
        1. 查询文件记录
        2. 删除本地文件
        3. 删除数据库记录
        """
        with Session() as session:
            # 查询文件记录
            file = session.query(File).filter(File.id == file_id).first()
            if not file:
                return False

            # 删除本地文件
            if os.path.exists(file.filepath):
                os.remove(file.filepath)
                logger.info(f"本地文件已删除: {file.filepath}")

            # 删除数据库记录
            session.delete(file)
            session.commit()
            logger.info(f"数据库记录已删除: file_id={file_id}")

        return True
