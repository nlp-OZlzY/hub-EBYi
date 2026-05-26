"""Milvus Vector Database Service with Embedding Support"""
from pymilvus import MilvusClient, DataType
import os
from services.embedding_service import EmbeddingService

class MilvusService:
    """Service class for Milvus vector database operations"""

    def __init__(self, uri: str = None, token: str = None):
        self.uri = uri or os.getenv("MILVUS_URI", "https://in03-e63b56293aba7fb.serverless.aws-eu-central-1.cloud.zilliz.com")
        self.token = token or os.getenv("MILVUS_TOKEN", "72746a502df7409ec9130f799fa678a4d0902a7be1b4ca9ae4093b656d9683eeacda15effe9378706ba2e4fb0a755a587ee011c7")
        self._client = None
        self._embedding_service = EmbeddingService()
        self.dim = 1152  # tongyi-embedding-vision-plus-2026-03-06 returns 1152-dim vectors

    @property
    def client(self):
        """Lazy initialization of Milvus client"""
        if self._client is None:
            self._client = MilvusClient(uri=self.uri, token=self.token, db_name="default")
        return self._client

    def _ensure_collection(self, collection_name: str):
        """Ensure collection exists, create if not"""
        try:
            collections = self.client.list_collections()
            if collection_name not in collections:
                schema = MilvusClient.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True,
                )
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=255)
                schema.add_field(field_name="page", datatype=DataType.INT64)
                schema.add_field(field_name="metadata", datatype=DataType.JSON)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="AUTOINDEX",
                    metric_type="COSINE"
                )

                self.client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params
                )
                print(f"[MilvusService] Created collection: {collection_name}")
        except Exception as e:
            print(f"[MilvusService] _ensure_collection error: {e}")

    def search_text(self, collection_name: str, query: str, limit: int = 5) -> list:
        """
        Search text chunks in Milvus.

        Args:
            collection_name: Name of the collection to search
            query: Text query string
            limit: Maximum number of results to return

        Returns:
            list: Search results with text and metadata
        """
        try:
            query_vector = self._embedding_service.embed(query)
            if not query_vector:
                return []

            self._ensure_collection(collection_name)
            self.client.load_collection(collection_name)
            
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=limit,
                search_params={"metric_type": "COSINE"},
                output_fields=["text", "document_id", "page", "metadata"]
            )
            return [
                {
                    "text": hit.get("entity", {}).get("text", ""),
                    "page": hit.get("entity", {}).get("page", 0),
                    "metadata": hit.get("entity", {}).get("metadata", {}),
                    "distance": hit.get("distance", 0)
                }
                for hit in results[0] if hit
            ]
        except Exception as e:
            print(f"[MilvusService] search_text error: {e}")
            return []

    def search_image(self, collection_name: str, query: str, limit: int = 3) -> list:
        """
        Search image chunks in Milvus.

        Args:
            collection_name: Name of the image collection
            query: Text query for image search
            limit: Maximum number of results

        Returns:
            list: Image results with metadata
        """
        try:
            query_vector = self._embedding_service.embed(query)
            if not query_vector:
                return []

            self._ensure_collection(collection_name)
            self.client.load_collection(collection_name)
            
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=limit,
                search_params={"metric_type": "COSINE"},
                output_fields=["text", "document_id", "page", "metadata"]
            )
            return [
                {
                    "image_base64": hit.get("entity", {}).get("image_base64", ""),
                    "page": hit.get("entity", {}).get("page", 0),
                    "metadata": hit.get("entity", {}).get("metadata", {})
                }
                for hit in results[0] if hit
            ]
        except Exception as e:
            print(f"[MilvusService] search_image error: {e}")
            return []

    def insert_text(self, collection_name: str, text: str, document_id: str, page: int, metadata: dict):
        """Insert text chunk into Milvus collection"""
        try:
            self._ensure_collection(collection_name)
            
            embedding = self._embedding_service.embed(text)
            if not embedding:
                print(f"[MilvusService] Failed to generate embedding for text")
                return

            self.client.insert(
                collection_name=collection_name,
                data=[{
                    "text": text,
                    "document_id": str(document_id),
                    "page": page,
                    "metadata": metadata,
                    "vector": embedding
                }]
            )
            self.client.flush([collection_name])
        except Exception as e:
            print(f"[MilvusService] insert_text error: {e}")

    def insert_image(self, collection_name: str, image_base64: str, document_id: str, page: int, metadata: dict):
        """Insert image into Milvus collection"""
        try:
            self._ensure_collection(collection_name)
            
            embedding = self._embedding_service.embed(image_base64[:1000])  # Use first 1000 chars for embedding
            if not embedding:
                print(f"[MilvusService] Failed to generate embedding for image")
                return

            self.client.insert(
                collection_name=collection_name,
                data=[{
                    "image_base64": image_base64,
                    "document_id": str(document_id),
                    "page": page,
                    "metadata": metadata,
                    "vector": embedding
                }]
            )
        except Exception as e:
            print(f"[MilvusService] insert_image error: {e}")