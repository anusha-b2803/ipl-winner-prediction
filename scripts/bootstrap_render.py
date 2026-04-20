import os
import sqlite3
import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bootstrap")

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ipl_stats.db")
YEARS = list(range(2008, 2027))

def is_db_empty():
    if not os.path.exists(DB_PATH):
        return True
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM team_season_stats")
        count = c.fetchone()[0]
        conn.close()
        return count == 0
    except Exception:
        return True

async def bootstrap():
    log.info("Checking if bootstrap is required...")
    if not is_db_empty():
        log.info("Database already contains data. Skipping bootstrap.")
        return

    log.info("🚀 EMPTY DATABASE DETECTED! Starting full historical bootstrap (2008-2026)...")
    log.info("This process may take 10-15 minutes but will only run once.")

    try:
        # 1. Step 1: Import Season Stats from Wikipedia (Populates SQLite)
        log.info("Step 1/3: Importing season stats from Wikipedia...")
        subprocess.run([sys.executable, "scripts/import_history.py"], check=True)

        # 2. Step 2: Scrape Match Results (Populates data/raw/*.json)
        log.info("Step 2/3: Scraping match results for all years...")
        # We run in chunks or one big call
        subprocess.run([sys.executable, "scraper/ipl_scraper.py", "--years"] + [str(y) for y in YEARS], check=True)

        # 3. Step 3: Ingest Matches and Embeddings (Populates Qdrant + SQLite matches)
        log.info("Step 3/3: Ingesting into Vector DB and SQLite...")
        subprocess.run([sys.executable, "pipeline/ingest.py", "--years"] + [str(y) for y in YEARS], check=True)

        log.info("✅ FULL BOOTSTRAP COMPLETED SUCCESSFULLY!")
    except Exception as e:
        log.error(f"❌ Bootstrap failed: {e}")
        # We don't exit(1) to let the API start anyway, though it will be empty
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(bootstrap())
