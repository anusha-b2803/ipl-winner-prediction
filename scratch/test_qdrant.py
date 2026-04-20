import os
import sys
from pathlib import Path

# Set environment variables BEFORE importing the module
test_path = "./data/test_vector_db"
if os.path.exists(test_path):
    import shutil
    try:
        shutil.rmtree(test_path)
    except Exception:
        pass

os.environ["QDRANT_PATH"] = test_path
os.environ["QDRANT_URL"] = "http://non-existent:6333"

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from rag.predictor import get_qdrant

def test_qdrant_init():
    print(f"Testing Qdrant initialization with path: {test_path}")
    client = get_qdrant()
    
    if client is not None and os.path.exists(test_path):
        print("SUCCESS: Qdrant initialized local storage at " + test_path)
    else:
        print("FAILURE: Qdrant did not initialize local storage as expected")

if __name__ == "__main__":
    test_qdrant_init()
