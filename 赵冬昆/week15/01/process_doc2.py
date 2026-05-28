
"""Process document 2"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker.process_document import process_document

print("=== Processing document 2 ===")

try:
    success = process_document(2)
    if success:
        print("Document processing successful")
    else:
        print("Document processing failed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
