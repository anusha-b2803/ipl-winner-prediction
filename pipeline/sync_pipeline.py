import asyncio
import logging
import datetime
import os
import sys
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.live_update import sync_live_data
from pipeline.ingest import main as ingest_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

async def run_sync():
    """
    Full sync pipeline: 
    1. Scrape latest matches and standings (SyncService)
    2. Embed and Ingest into Qdrant + SQLite
    """
    current_year = datetime.datetime.now().year
    log.info(f"🚀 Starting automated sync for IPL {current_year}...")
    
    try:
        # 1. Sync live data (matches + standings) to data/raw/
        await sync_live_data()
        
        # 2. Re-use existing Qdrant client if available in the process to avoid locking issues
        from rag.predictor import get_qdrant as get_active_qdrant
        active_client = get_active_qdrant()
        
        # 3. Run ingestion for the current year
        await ingest_main(years=[current_year], shared_qdrant=active_client)
        
        log.info("✅ Automated sync completed successfully.")
        return True
    except Exception as e:
        log.error(f"❌ Automated sync failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(run_sync())
