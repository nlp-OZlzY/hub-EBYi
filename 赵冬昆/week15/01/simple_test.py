
# Simple test script
print("Starting test...")

try:
    from services.milvus_service import MilvusService
    print("Import successful")
    
    m = MilvusService()
    print("MilvusService created")
    
    cols = m.client.list_collections()
    print(f"Collections: {cols}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
