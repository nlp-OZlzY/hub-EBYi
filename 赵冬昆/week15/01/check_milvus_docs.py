
"""Check documents in Milvus"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus = MilvusService()
client = milvus.client

# Check if collection exists
cols = client.list_collections()
print(f"Collections: {cols}")

if 'documents' in cols:
    print("\nDocuments collection exists")
    
    # Get stats
    try:
        stats = client.get_collection_stats('documents')
        print(f"Stats: {stats}")
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    # Query data
    print("\nQuerying data...")
    try:
        results = client.query(
            collection_name='documents',
            filter="",
            limit=5,
            output_fields=["text", "page"]
        )
        print(f"Found {len(results)} entities")
        for r in results:
            print(f"\nPage {r.get('page')}:")
            print(f"Text: {r.get('text', '')[:100]}...")
    except Exception as e:
        print(f"Error querying: {e}")
else:
    print("\nDocuments collection does not exist")
