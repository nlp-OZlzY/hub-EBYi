"""Qiniu Cloud Storage Service"""
from qiniu import Auth, put_file, etag
import qiniu.config
import os

class QiniuService:
    """Service class for Qiniu Cloud Storage operations"""

    def __init__(self):
        self.access_key = os.getenv("QINIU_ACCESS_KEY", "fmelP1qnrKlVuAGAdcRXVPb-eBSgcKz8kDDRjbQ6")
        self.secret_key = os.getenv("QINIU_SECRET_KEY", "JI0tlniy62avkHU0aN9iiSDZeJs41m2mKaYJHJ-C")
        self.bucket_name = os.getenv("QINIU_BUCKET", "duixiangcunchu001")
        self.q = Auth(self.access_key, self.secret_key)

    def upload_file(self, local_file_path: str, key_in_qiniu: str = None) -> str:
        """
        Upload a file to Qiniu storage.

        Args:
            local_file_path: Path to local file
            key_in_qiniu: Key (filename) in Qiniu, None for auto-generated

        Returns:
            str: URL of uploaded file in Qiniu
        """
        token = self.q.upload_token(self.bucket_name, key=key_in_qiniu)
        ret, info = put_file(token, key_in_qiniu, local_file_path)
        if ret:
            base_url = f"https://{self.bucket_name}.qiniu.com"
            if key_in_qiniu:
                return f"{base_url}/{key_in_qiniu}"
            return f"{base_url}/{ret.get('key', '')}"
        raise Exception(f"Qiniu upload failed: {info}")

    def get_file_url(self, key: str) -> str:
        """Get the URL for a file in Qiniu"""
        base_url = f"https://{self.bucket_name}.qiniu.com"
        return f"{base_url}/{key}"