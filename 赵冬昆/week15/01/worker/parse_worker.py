"""Kafka Consumer Worker for Document Parsing via MinerU"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
from datetime import datetime
from typing import Optional

from kafka import KafkaConsumer, KafkaProducer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.orm_model import Base, Document, DocumentStatus
from services.mineru_service import MinerUService
from services.milvus_service import MilvusService


class ParseWorker:
    """Background worker that consumes Kafka messages and processes documents via MinerU"""

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "document_parse_topic",
        kafka_group_id: str = "document_parse_group",
        milvus_uri: str = "http://localhost:19530",
        milvus_token: str = "root:Milvus",
        db_path: str = "sqlite:///./multimodal_rag.db"
    ):
        self.kafka_topic = kafka_topic
        self.milvus_service = MilvusService(uri=milvus_uri, token=milvus_token)
        self.mineru_service = MinerUService()

        self.engine = create_engine(db_path, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=kafka_bootstrap_servers.split(","),
            group_id=kafka_group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True
        )

    def process_message(self, message: dict) -> bool:
        """
        Process a single Kafka message and parse the document via MinerU.

        Args:
            message: dict containing document_id and qiniu_url

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        document_id = message.get("document_id")
        qiniu_url = message.get("qiniu_url")

        if not document_id:
            return False

        session = self.SessionLocal()
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False

            document.status = DocumentStatus.PROCESSING
            session.commit()

            local_path = document.file_path
            print(f"Processing document locally: {local_path}")
            
            if os.path.exists(local_path):
                result = self.mineru_service.parse_document(local_path)
            else:
                print(f"Local file not found: {local_path}")
                return False

            if result.get("success"):
                chunks = result.get("chunks", [])
                images = result.get("images", [])

                for chunk in chunks:
                    self.milvus_service.insert_text(
                        collection_name="documents",
                        text=chunk["content"],
                        document_id=str(document_id),
                        page=chunk.get("page", 0),
                        metadata={
                            "file_name": document.file_name,
                            "chunk_id": chunk.get("id", str(uuid.uuid4()))
                        }
                    )

                for image in images:
                    self.milvus_service.insert_image(
                        collection_name="document_images",
                        image_base64=image.get("base64"),
                        document_id=str(document_id),
                        page=image.get("page", 0),
                        metadata={
                            "file_name": document.file_name,
                            "image_id": image.get("id", str(uuid.uuid4()))
                        }
                    )

                document.status = DocumentStatus.COMPLETED
                document.page_count = len(chunks)
            else:
                document.status = DocumentStatus.FAILED

            session.commit()
            return True

        except Exception as e:
            session.rollback()
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = DocumentStatus.FAILED
                session.commit()
            return False
        finally:
            session.close()

    def start(self):
        """Start consuming messages from Kafka"""
        print(f"[ParseWorker] Starting consumer on topic: {self.kafka_topic}")
        for message in self.consumer:
            try:
                print(f"[ParseWorker] Received message: {message.value}")
                self.process_message(message.value)
            except Exception as e:
                print(f"[ParseWorker] Error processing message: {e}")

    def stop(self):
        """Stop the consumer and close connections"""
        if self.consumer:
            self.consumer.close()
        if self.engine:
            self.engine.dispose()


if __name__ == "__main__":
    try:
        worker = ParseWorker()
        print("[ParseWorker] Worker started successfully")
        worker.start()
    except KeyboardInterrupt:
        print("[ParseWorker] Worker stopped by user")
        worker.stop()
    except Exception as e:
        print(f"[ParseWorker] Error starting worker: {e}")
        import traceback
        traceback.print_exc()


