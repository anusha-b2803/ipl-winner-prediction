from qdrant_client import QdrantClient
import os

QDRANT_PATH = './data/vector_db_v3'
COLLECTION_NAME = 'ipl_matches'

def check_rag():
    if not os.path.exists(QDRANT_PATH):
        print(f"FAILED: Qdrant path {QDRANT_PATH} does not exist.")
        return

    try:
        client = QdrantClient(path=QDRANT_PATH)
        collections = [c.name for c in client.get_collections().collections]
        print(f"Collections found: {collections}")
        
        if COLLECTION_NAME in collections:
            count = client.count(COLLECTION_NAME).count
            print(f"SUCCESS: Collection '{COLLECTION_NAME}' exists with {count} points.")
            
            # Check for 2026 data
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            count_2026 = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must=[FieldCondition(key="year", match=MatchValue(value=2026))])
            ).count
            print(f"RAG Context: Found {count_2026} records for the current 2026 season.")
        else:
            print(f"FAILED: Collection '{COLLECTION_NAME}' not found.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    check_rag()
