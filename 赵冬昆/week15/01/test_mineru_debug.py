import sys
sys.path.insert(0, '.')

from services.mineru_service import MinerUService

mineru = MinerUService()
pdf_file = './uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'
result = mineru.parse_document(pdf_file)
print("\n=== FINAL RESULT ===")
print(result)