"""Milvus Vector Store Service."""
import os
import yaml
from typing import List, Dict, Any

from pymilvus import MilvusClient

_config = None


def load_config():
    global _config
    if _config is None:
        cfg_path = "config.yaml"
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(cfg_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


class VectorStore:
    def __init__(self):
        cfg = load_config()
        mc = cfg["milvus"]

        self.client = MilvusClient(
            uri=mc["uri"],
            token=mc["token"]
        )
        self.collection = mc["collection"]
        self._ensure_collection()

    def _ensure_collection(self):
        """确保collection存在"""
        if not self.client.has_collection(self.collection):
            # 创建collection和index
            self.client.create_collection(
                collection_name=self.collection,
                dimension=512,
                metric_type="IP"
            )

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_expr: str = None
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=limit,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"],
            filter=filter_expr
        )
        return results[0] if results else []

    def insert(self, data: List[Dict[str, Any]]):
        """插入向量数据"""
        self.client.insert(
            collection_name=self.collection,
            data=data
        )

    def delete(self, filter_expr: str):
        """删除向量数据"""
        self.client.delete(
            collection_name=self.collection,
            filter=filter_expr
        )


_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store