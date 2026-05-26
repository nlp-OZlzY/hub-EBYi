import sys
sys.path.insert(0, '.')

from services.mineru_service import MinerUService

mineru = MinerUService()
pdf_file = './uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'
print(f"Testing PDF parsing on: {pdf_file}\n")

result = mineru.parse_document(pdf_file)

print("\n=== FINAL RESULT ===")
print(f"Success: {result.get('success')}")
if result.get('success'):
    print(f"Chunks: {len(result.get('chunks', []))}")
    print(f"First chunk preview: {result.get('chunks', [{}])[0].get('content', '')[:200]}...")
else:
    print(f"Error: {result.get('error')}")