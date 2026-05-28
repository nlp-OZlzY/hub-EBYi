"""向量检索服务"""
import logging
from typing import List, Dict
from pymilvus import MilvusClient
from config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """向量检索服务"""

    def __init__(self):
        self.client = MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN
        )
        self.collection_name = settings.MILVUS_COLLECTION

    def search(self, query_vector: list, limit: int = 5) -> List[Dict]:
        """
        向量检索

        Args:
            query_vector: 查询向量
            limit: 返回结果数量

        Returns:
            检索结果列表
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=limit,
                anns_field="text_vector",
                output_fields=["text", "db_id", "file_name", "file_path"]
            )

            # 解析结果
            parsed_results = []
            for result in results[0]:
                parsed_results.append({
                    "text": result["entity"]["text"],
                    "db_id": result["entity"]["db_id"],
                    "file_name": result["entity"]["file_name"],
                    "file_path": result["entity"]["file_path"],
                    "distance": result["distance"]
                })

            return parsed_results

        except Exception as e:
            logger.error(f"Milvus检索失败: {e}")
            return []

    def delete_by_file_id(self, file_id: int):
        """
        删除指定文件的所有向量

        Args:
            file_id: 文件ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"db_id == {file_id}"
            )
            logger.info(f"Milvus删除完成: db_id={file_id}")
        except Exception as e:
            logger.error(f"Milvus删除失败: {e}")
