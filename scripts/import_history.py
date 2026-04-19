import urllib.request
import pandas as pd
import sqlite3
import os
import re
import json
import datetime
from pathlib import Path
from bs4 import BeautifulSoup

DB_PATH = 'data/ipl_stats.db'
RAW_DATA_DIR = Path('data/raw')
if not os.path.exists(os.path.dirname(DB_PATH)):
    os.makedirs(os.path.dirname(DB_PATH))
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_team_name(name):
    # Remove references like [a] or (C) or (R)
    name = re.sub(r'\[.*?\]', '', str(name))
    name = re.sub(r'\(.*?\)', '', name)
    return name.strip()

def extract_table_from_wiki(year):
    url = f"https://en.wikipedia.org/wiki/{year}_Indian_Premier_League"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read()
        tables = pd.read_html(html)
        
        # Find the points table
        for df in tables:
            # Check if it looks like a standings table
            cols = [str(c).lower() for c in df.columns]
            # Standings table normally has Team, Pts, NRR, and is somewhat small (~8-10 rows)
            if any('team' in c for c in cols) and any('pts' in c or 'points' in c for c in cols) and len(df) >= 8 and len(df) <= 12:
                # Rename columns dynamically to standard formats
                df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
                
                # Map various headers to our standard DB names
                col_map = {}
                for col in df.columns:
                    if 'team' in col: col_map[col] = 'team'
                    elif 'pld' in col or 'match' in col or 'm' == col: col_map[col] = 'matches'
                    elif col == 'w' or 'won' in col: col_map[col] = 'wins'
                    elif col == 'l' or 'lost' in col: col_map[col] = 'losses'
                    elif 'nr' in col and 'nrr' not in col: col_map[col] = 'no_result'
                    elif 't' == col or 'tied' in col: col_map[col] = 'no_result' # Combine tied into no_result for simplicity if no specific tied column
                    elif 'pts' in col or 'point' in col: col_map[col] = 'points'
                    elif 'nrr' in col or 'net run rate' in col: col_map[col] = 'nrr'
                df.rename(columns=col_map, inplace=True)
                
                # Clean team names
                df['team'] = df['team'].apply(clean_team_name)
                
                # Convert numeric cols
                for num_col in ['matches', 'wins', 'losses', 'no_result', 'points', 'nrr']:
                    if num_col in df.columns:
                        if num_col == 'nrr':
                            # Handle different minus signs and non-numeric junk
                            df[num_col] = df[num_col].astype(str).str.replace('−', '-').str.extract(r'([-+]?\d+\.\d+)').astype(float)
                        else:
                            df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0).astype(int)
                    else:
                        if num_col == 'nrr': df['nrr'] = 0.0
                        else: df[num_col] = 0
                
                # Determine position
                df['position'] = range(1, len(df) + 1)
                
                return df[['team', 'matches', 'wins', 'losses', 'no_result', 'points', 'nrr', 'position']]
        print(f"[{year}] Warning: Points table format unrecognized.")
        return None
    except Exception as e:
        print(f"[{year}] Error scraping Wikipedia: {e}")
        return None

def save_to_json(year, df, winner=None):
    """Save stats to JSON file for persistence and user request."""
    data = {
        "year": year,
        "winner": winner or "TBD",
        "team_stats": df.to_dict(orient='records')
    }
    file_path = RAW_DATA_DIR / f"ipl_{year}.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return file_path

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS team_season_stats (
                year INTEGER, team TEXT, matches INTEGER, wins INTEGER, losses INTEGER,
                no_result INTEGER, points INTEGER, nrr REAL, position INTEGER,
                qualified_playoffs BOOLEAN, won_title BOOLEAN,
                UNIQUE(year, team)
             )''')
             
    # Known winners override
    winners = {
        2008: "Rajasthan Royals", 2009: "Deccan Chargers", 2010: "Chennai Super Kings", 2011: "Chennai Super Kings",
        2012: "Kolkata Knight Riders", 2013: "Mumbai Indians", 2014: "Kolkata Knight Riders", 2015: "Mumbai Indians",
        2016: "Sunrisers Hyderabad", 2017: "Mumbai Indians", 2018: "Chennai Super Kings", 2019: "Mumbai Indians",
        2020: "Mumbai Indians", 2021: "Chennai Super Kings", 2022: "Gujarat Titans", 2023: "Chennai Super Kings",
        2024: "Kolkata Knight Riders", 2025: "Royal Challengers Bengaluru"
    }

    current_year = datetime.datetime.now().year
    
    for year in range(2008, current_year + 1):
        print(f"[{year}] Updating data...", end=" ", flush=True)
        df = extract_table_from_wiki(year)
        if df is not None:
            winner = winners.get(year)
            # Save to JSON
            save_to_json(year, df, winner)
            
            records_added = 0
            for idx, row in df.iterrows():
                team_name = row['team']
                if 'TBD' in team_name or not team_name:
                    continue
                    
                is_playoff = bool(row['position'] <= 4)
                won_title = bool(winner == team_name)
                
                c.execute('''
                    INSERT INTO team_season_stats 
                    (year, team, matches, wins, losses, no_result, points, nrr, position, qualified_playoffs, won_title)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(year, team) DO UPDATE SET
                    matches=excluded.matches, wins=excluded.wins, losses=excluded.losses, points=excluded.points, 
                    no_result=excluded.no_result, nrr=excluded.nrr, position=excluded.position, 
                    qualified_playoffs=excluded.qualified_playoffs, won_title=excluded.won_title
                ''', (year, team_name, int(row['matches']), int(row['wins']), int(row['losses']), int(row['no_result']), 
                      int(row['points']), float(row['nrr']), int(row['position']), is_playoff, won_title))
                records_added += 1
            conn.commit()
            print(f"OK: {records_added} teams updated.")
        else:
            print("FAILED")

    conn.close()
    print("✅ All years synchronized and exported to data/raw/")

if __name__ == '__main__':
    main()
