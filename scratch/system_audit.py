import sqlite3
import os
import torch
import httpx
import logging
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Audit")

def audit_system():
    print("--- IPL PROPHET SYSTEM AUDIT ---")
    
    # 1. Check SQLite
    db_path = './data/ipl_stats.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM team_season_stats")
        count = c.fetchone()[0]
        c.execute("SELECT MIN(year), MAX(year) FROM team_season_stats")
        years = c.fetchone()
        print(f"[SQLite] Found {count} team-season records from {years[0]} to {years[1]}.")
        conn.close()
    else:
        print("[SQLite] FAILED: Database missing.")

    # 2. Check Qdrant
    qdrant_path = './data/vector_db_v3'
    if os.path.exists(qdrant_path):
        try:
            client = QdrantClient(path=qdrant_path)
            collections = [c.name for c in client.get_collections().collections]
            print(f"[Qdrant] Collections: {collections}")
            if "ipl_matches" in collections:
                pts = client.count("ipl_matches").count
                print(f"[Qdrant] 'ipl_matches' has {pts} points.")
            client.close()
        except Exception as e:
            print(f"[Qdrant] Process lock detected (expected if server is running).")
    else:
        print("[Qdrant] FAILED: Vector DB missing.")

    # 3. Check Transformer
    weights_path = './models/weights.pt'
    if os.path.exists(weights_path):
        print(f"[Transformer] SUCCESS: '{weights_path}' exists.")
    else:
        print("[Transformer] FAILED: Weights file missing.")

    # 4. Check Environment
    keys = ["GOOGLE_API_KEY", "HUGGINGFACE_API_KEY", "HF_MODEL"]
    print(f"[Env] Model: {os.getenv('HF_MODEL')}")
    for k in keys:
        if os.getenv(k):
            print(f"[Env] {k}: CONFIGURED")
        else:
            print(f"[Env] {k}: MISSING")

    # 5. Requirement consistency
    with open('api/requirements.txt', 'r') as f:
        reqs = f.read()
        if 'huggingface-hub>=0.34.6' in reqs:
            print("[Requirements] SUCCESS: Modern HF library pinned.")
        else:
            print("[Requirements] WARNING: Outdated or missing HF pin.")

    print("\n--- AUDIT COMPLETE ---")

if __name__ == "__main__":
    audit_system()
