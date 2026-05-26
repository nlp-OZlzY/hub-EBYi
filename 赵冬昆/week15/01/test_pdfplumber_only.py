import sys
sys.path.insert(0, '.')

# 直接测试 pdfplumber fallback
from services.mineru_service import MinerUService

mineru = MinerUService()
pdf_file = './uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'

# 直接调用 pdfplumber
result = mineru._parse_with_pdfplumber(pdf_file)

print("=== PDF PLUMBER RESULT ===")
print(f"Success: {result.get('success')}")
if result.get('success'):
    chunks = result.get('chunks', [])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk (page {chunks[0].get('page')}): {chunks[0].get('content')[:200]}...")
        print(f"Last chunk (page {chunks[-1].get('page')}): {chunks[-1].get('content')[:200]}...")
else:
    print(f"Error: {result.get('error')}")