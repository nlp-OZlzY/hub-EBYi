"""MinerU Service for document parsing - using pdfplumber only"""
import os
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

class MinerUService:
    """Service class for PDF document parsing using pdfplumber"""

    def __init__(self):
        logger.info("MinerUService initialized (using pdfplumber)")

    def parse_document(self, pdf_path: str) -> dict:
        """
        Parse a document using pdfplumber.

        Args:
            pdf_path: Path to the PDF document (local file path)

        Returns:
            dict with parsing result containing chunks and images
        """
        logger.info(f"=== Starting PDF parsing ===")
        logger.info(f"PDF path: {pdf_path}")

        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return {"success": False, "error": f"File not found: {pdf_path}"}

        pdf_abs_path = os.path.abspath(pdf_path)
        logger.info(f"Absolute PDF path: {pdf_abs_path}")
        logger.info(f"File exists: {os.path.exists(pdf_abs_path)}")
        logger.info(f"File size: {os.path.getsize(pdf_abs_path)} bytes")

        return self._parse_with_pdfplumber(pdf_abs_path)

    def _parse_with_pdfplumber(self, pdf_abs_path: str) -> dict:
        """Parse PDF using pdfplumber"""
        try:
            import pdfplumber

            logger.info("Using pdfplumber for PDF parsing")
            chunks = []

            with pdfplumber.open(pdf_abs_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            "content": text,
                            "page": page_num + 1
                        })
                        if page_num < 5 or page_num % 50 == 0:
                            logger.info(f"Extracted {len(text)} chars from page {page_num + 1}")

            logger.info(f"Total chunks from pdfplumber: {len(chunks)}")

            if not chunks:
                return {"success": False, "error": "No text extracted from PDF"}

            return {
                "success": True,
                "data": {"chunks": chunks},
                "chunks": chunks,
                "images": []
            }

        except ImportError:
            logger.error("pdfplumber not installed")
            return {"success": False, "error": "pdfplumber not installed"}
        except Exception as e:
            logger.exception(f"pdfplumber exception: {e}")
            return {"success": False, "error": str(e)}