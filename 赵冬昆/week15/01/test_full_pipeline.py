
"""Test full document processing pipeline"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker.process_document import process_document, list_documents

print("=== 开始测试完整文档处理流程 ===")

# List documents first
list_documents()

# Process document 2
print("\n=== 处理文档 2 ===")
success = process_document(2)

if success:
    print("\n✅ 文档处理成功！")
else:
    print("\n❌ 文档处理失败！")

# List documents again
print("\n=== 更新后的文档列表 ===")
list_documents()
