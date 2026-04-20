import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from scraper.ipl_scraper import scrape_series_matches, MatchResult
import httpx

async def test_scraper_fallback():
    async with httpx.AsyncClient() as client:
        # Test with a non-existent series ID to force 404/failure
        print("Testing scraper fallback with invalid year (forcing failure)...")
        # We manually trigger a case where fetch fails or series ID is missing
        # In our code, scrape_series_matches uses IPL_SERIES.get(year)
        # 1999 is not in IPL_SERIES
        matches = await scrape_series_matches(client, 1999)
        
        # Actually scrape_series_matches returns [] if series_id is not found at line 136
        # Let's test a year that IS in IPL_SERIES but we force fetch to fail
        # Actually, let's just test 2026. If it 404s, it should use fallback.
        print("Testing scraper for 2026...")
        matches = await scrape_series_matches(client, 2026)
        
        if len(matches) > 0:
            print(f"SUCCESS: Got {len(matches)} matches (could be real or fallback)")
            print(f"First match: {matches[0].team1} vs {matches[0].team2}")
        else:
            print("FAILURE: Got 0 matches")

if __name__ == "__main__":
    asyncio.run(test_scraper_fallback())
