import sqlite3
import os

DB_PATH = 'data/ipl_stats.db'
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM team_season_stats WHERE team LIKE '%TBD%' OR team = ''")
    c.execute("DELETE FROM match_results WHERE winner LIKE '%TBD%' OR winner = ''")
    conn.commit()
    print("Cleaned up TBD from database")
    
    # Test checking 2026
    c.execute("SELECT team, wins FROM team_season_stats WHERE year=2026 ORDER BY position ASC")
    print(c.fetchall())
    
    # Check what years exist
    c.execute("SELECT DISTINCT year FROM team_season_stats ORDER BY year DESC")
    print(c.fetchall())
    
    conn.close()
