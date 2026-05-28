
"""Check Milvus data"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()
client = milvus_service.client

# List collections
print("Collections:", client.list_collections())

# Check if documents collection exists
if 'documents' in client.list_collections():
    # Get collection info
    info = client.get_collection_info('documents')
    print("\nCollection info:", info)
    
    # Get stats
    stats = client.get_collection_stats('documents')
    print("\nCollection stats:", stats)
    
    # Query some data
    print("\nQuerying first 5 entities...")
    results = client.query(
        collection_name='documents',
        filter="",
        limit=5,
        output_fields=["text", "document_id", "page", "metadata"]
    )
    print(f"Found {len(results)} entities")
    for i, r in enumerate(results):
        print(f"\nEntity {i+1}:")
        print(f"  ID: {r.get('id')}")
        print(f"  Text: {r.get('text', '')[:100]}...")
        print(f"  Document ID: {r.get('document_id')}")
        print(f"  Page: {r.get('page')}")
        print(f"  Metadata: {r.get('metadata')}")
else:
    print("\n'documents' collection does not exist!")
