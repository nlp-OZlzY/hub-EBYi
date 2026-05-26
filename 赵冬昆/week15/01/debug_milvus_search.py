
"""Debug Milvus search results structure"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_service import MilvusService

milvus_service = MilvusService()

# Direct search to see raw results
client = milvus_service.client

# Get embedding for query
query_text = "汽车发动机"
query_vector = milvus_service._embedding_service.embed(query_text)
print(f"Query vector length: {len(query_vector)}")

# Raw search
print("\nRaw search results:")
results = client.search(
    collection_name='documents',
    data=[query_vector],
    limit=5,
    search_params={"metric_type": "COSINE"}
)

print(f"Results type: {type(results)}")
print(f"Results length: {len(results)}")

if len(results) > 0:
    print(f"\nFirst result type: {type(results[0])}")
    print(f"First result length: {len(results[0])}")
    
    if len(results[0]) > 0:
        hit = results[0][0]
        print(f"\nHit type: {type(hit)}")
        print(f"Hit keys: {hit.keys() if hasattr(hit, 'keys') else 'N/A'}")
        
        if hasattr(hit, 'keys'):
            for key in hit.keys():
                print(f"\n  {key}: {type(hit[key])}")
                if key == 'entity':
                    print(f"    Entity content: {hit[key]}")
                elif key == 'distance':
                    print(f"    Distance: {hit[key]}")

# Test the search_text method
print("\n\nTesting search_text method:")
text_results = milvus_service.search_text('documents', '汽车发动机', limit=3)
print(f"search_text results: {text_results}")
