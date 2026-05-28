
"""Deep debug search issue"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService
from services.embedding_service import EmbeddingService

print("=== Deep Debug ===")

# Initialize
milvus = MilvusService()
embed = EmbeddingService()
client = milvus.client

# Step 1: Check collection
print("\n[1] Checking collection...")
cols = client.list_collections()
print(f"Collections: {cols}")

if 'documents' in cols:
    client.load_collection('documents')
    stats = client.get_collection_stats('documents')
    print(f"Documents stats: {stats}")
    
    # Query data
    print("\n[2] Querying data...")
    results = client.query(
        collection_name='documents',
        filter="",
        limit=3,
        output_fields=["text", "page", "document_id"]
    )
    print(f"Query found {len(results)} entities")
    for r in results:
        print(f"  Page {r.get('page')}: {r.get('text')[:50]}...")
    
    # Step 3: Test search step by step
    print("\n[3] Testing search step by step...")
    
    # Get query vector
    query_text = "发动机"
    query_vector = embed.embed(query_text)
    print(f"Query: {query_text}")
    print(f"Vector length: {len(query_vector)}")
    print(f"Vector first 5 values: {query_vector[:5]}")
    
    # Raw search
    print("\nRaw search...")
    raw_results = client.search(
        collection_name='documents',
        data=[query_vector],
        limit=3,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "page"]
    )
    
    print(f"Raw results length: {len(raw_results)}")
    if len(raw_results) > 0:
        print(f"First result length: {len(raw_results[0])}")
        
        for hit in raw_results[0]:
            print(f"\nHit:")
            print(f"  Distance: {hit.get('distance')}")
            print(f"  ID: {hit.get('id')}")
            print(f"  Entity type: {type(hit.get('entity'))}")
            entity = hit.get('entity')
            if entity:
                print(f"  Entity keys: {entity.keys()}")
                print(f"  Text: {entity.get('text', '')[:50]}...")
            else:
                print("  Entity is None!")
else:
    print("documents collection does not exist")

print("\n=== Debug Complete ===")
