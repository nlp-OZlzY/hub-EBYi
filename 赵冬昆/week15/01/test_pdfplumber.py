"""Simple PDF text extraction as fallback when MinerU CLI doesn't work"""
import pdfplumber
import os

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extract text from PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        list of dicts with text and page info
    """
    if not os.path.exists(pdf_path):
        return []

    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "content": text,
                        "page": page_num + 1
                    })
                    print(f"Extracted {len(text)} chars from page {page_num + 1}")
    except Exception as e:
        print(f"Error extracting text: {e}")

    return chunks

if __name__ == "__main__":
    pdf_file = './uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'
    print(f"Testing PDF text extraction on: {pdf_file}")
    chunks = extract_text_from_pdf(pdf_file)
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk preview: {chunks[0]['content'][:200]}...")