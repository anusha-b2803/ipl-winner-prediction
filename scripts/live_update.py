import sqlite3
import datetime
import os
import sys
import json
import asyncio
from pathlib import Path
from dataclasses import asdict

# Add project root to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.import_history import extract_table_from_wiki, DB_PATH, RAW_DATA_DIR
from scraper.ipl_scraper import scrape_series_matches, IPL_WINNERS
import httpx

async def sync_live_data():
    """
    Hybrid Sync: 
    - Standings from Wikipedia (usually fast/accurate for table)
    - Match results from ESPNcricinfo (detailed MatchCards)
    """
    current_year = datetime.datetime.now().year
    print(f"[{datetime.datetime.now()}] Starting live sync for IPL {current_year}...")
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Scrape Standings
        # Note: We use pandas-based wiki scraper, but we wrap it in a thread if needed
        # since it's blocking. For now, running directly.
        df = extract_table_from_wiki(current_year)
        
        # 2. Scrape Match Results
        matches = await scrape_series_matches(client, current_year)
        
    if df is not None:
        # Save combined data to JSON
        stats_list = df.to_dict(orient='records')
        data = {
            "year": current_year,
            "winner": IPL_WINNERS.get(current_year, "TBD"),
            "matches": [asdict(m) for m in matches],
            "team_stats": stats_list
        }
        
        file_path = RAW_DATA_DIR / f"ipl_{current_year}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully synced {len(stats_list)} teams and {len(matches)} matches for IPL {current_year}.")
        return True
    else:
        print(f"Failed to fetch standings data for IPL {current_year}.")
        return False

# Alias for backward compatibility with API
update_current_season = sync_live_data

if __name__ == "__main__":
    asyncio.run(sync_live_data())
