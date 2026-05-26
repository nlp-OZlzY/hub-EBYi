
"""Simple search test"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus = MilvusService()
client = milvus.client

# Check if documents collection exists
cols = client.list_collections()
print("Collections:", cols)

if 'documents' in cols:
    # Load collection
    client.load_collection('documents')
    
    # Get stats
    stats = client.get_collection_stats('documents')
    print("Stats:", stats)
    
    # Test query
    print("\nQuery test:")
    results = client.query(
        collection_name='documents',
        filter="",
        limit=2,
        output_fields=["text", "page"]
    )
    print(f"Query results: {len(results)}")
    for r in results:
        print(f"Page {r.get('page')}: {r.get('text')[:50]}...")
    
    # Test search
    print("\nSearch test:")
    from services.embedding_service import EmbeddingService
    embed = EmbeddingService()
    
    query_vector = embed.embed("发动机")
    print(f"Query vector length: {len(query_vector)}")
    
    search_results = client.search(
        collection_name='documents',
        data=[query_vector],
        limit=2,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "page"]
    )
    
    print(f"Search results count: {len(search_results)}")
    if len(search_results) > 0:
        for hit in search_results[0]:
            print(f"\nHit:")
            print(f"  Distance: {hit.get('distance')}")
            print(f"  Entity: {hit.get('entity')}")
else:
    print("documents collection does not exist")
