"""
Data Pipeline
Reads scraped JSON → embeds match summaries → stores in Qdrant + SQLite.
Run: python pipeline/ingest.py --years 2023 2024
"""

import asyncio
import json
import logging
import argparse
import os
from pathlib import Path
from typing import Optional

import asyncpg  # Keep for typing if needed, but switching to aiosqlite
import aiosqlite
import httpx
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/vector_db")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ipl_stats.db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ipl_matches")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # FastEmbed bge-small-en-v1.5 dimension

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ─── Embedding (Local FastEmbed) ──────────────────────────────────────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        log.info(f"Initializing Local Embedding Model: {EMBEDDING_MODEL_NAME}...")
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embed_model

async def get_embedding(text: str) -> list[float]:
    """Get embedding for a text. Prefers Google API (Zero RAM) over local FastEmbed."""
    if GOOGLE_API_KEY:
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            log.warning(f"Google Embedding API failed, falling back to local: {e}")

    # Fallback to local
    model = get_embed_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed texts. Prefers Google API over local."""
    if GOOGLE_API_KEY:
        try:
            # Google allows batches of up to 100
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(result['embedding'])
            return all_embeddings
        except Exception as e:
            log.warning(f"Google Batch Embedding failed, falling back to local: {e}")

    # Fallback to local
    model = get_embed_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def build_rich_summary(match: dict, year: int) -> str:
    """Build a rich text summary for embedding."""
    summary = match.get("summary", "")
    winner = match.get("winner", "")
    margin = match.get("margin", "")
    venue = match.get("venue", "")
    team1 = match.get("team1", "")
    team2 = match.get("team2", "")
    team1_score = match.get("team1_score", "")
    team2_score = match.get("team2_score", "")

    return (
        f"IPL {year} match: {team1} vs {team2} at {venue}. "
        f"{team1} scored {team1_score}, {team2} scored {team2_score}. "
        f"Winner: {winner}" + (f" by {margin}" if margin else "") + ". "
        f"{summary}"
    )


def build_season_narrative(data: dict) -> str:
    """Build a season-level narrative for embedding."""
    year = data["year"]
    winner = data["winner"]
    teams = data.get("team_stats", [])

    top4 = [t["team"] for t in teams[:4]] if teams else []
    top_team = teams[0] if teams else {}

    return (
        f"IPL {year} season summary. "
        f"The {winner} won the IPL {year} title. "
        f"Top 4 playoff teams: {', '.join(top4)}. "
        f"League toppers: {top_team.get('team', '')} with "
        f"{top_team.get('wins', '')} wins and NRR of {top_team.get('nrr', '')}. "
        f"Total matches in the season: {len(data.get('matches', []))}."
    )


def build_h2h_summary(team1: str, team2: str, matches: list[dict]) -> str:
    """Build head-to-head summary text."""
    h2h = [m for m in matches if
           (m["team1"] == team1 and m["team2"] == team2) or
           (m["team1"] == team2 and m["team2"] == team1)]
    t1_wins = sum(1 for m in h2h if m["winner"] == team1)
    t2_wins = sum(1 for m in h2h if m["winner"] == team2)
    return (
        f"Head to head: {team1} vs {team2}. "
        f"{team1} won {t1_wins} times, {team2} won {t2_wins} times out of {len(h2h)} matches."
    )


# ─── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant(existing_client: Optional[QdrantClient] = None) -> QdrantClient:
    """Get a Qdrant client, reusing an existing one if provided."""
    if existing_client is not None:
        return existing_client
        
    # Use local path if provided, otherwise fallback to URL
    if QDRANT_PATH:
        try:
            parent = os.path.dirname(QDRANT_PATH)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        return QdrantClient(path=QDRANT_PATH)
    
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_collection(qdrant: QdrantClient):
    existing = [c.name for c in qdrant.get_collections().collections]
    
    # Check for dimension mismatch
    if COLLECTION_NAME in existing:
        info = qdrant.get_collection(COLLECTION_NAME)
        current_dim = info.config.params.vectors.size
        if current_dim != EMBEDDING_DIM:
            log.warning(f"Dimension mismatch (expected {EMBEDDING_DIM}, found {current_dim}). Recreating collection...")
            qdrant.delete_collection(COLLECTION_NAME)
            existing.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info(f"Created Qdrant collection: {COLLECTION_NAME}")


async def upsert_to_qdrant(
    qdrant: QdrantClient,
    year: int,
    data: dict,
):
    texts = []
    payloads = []
    ids = []

    # Season narrative
    season_text = build_season_narrative(data)
    texts.append(season_text)
    payloads.append({
        "type": "season_summary",
        "year": year,
        "winner": data["winner"],
        "text": season_text,
    })
    ids.append(year * 100000)

    # Individual matches
    for i, match in enumerate(data.get("matches", []), 1):
        text = build_rich_summary(match, year)
        texts.append(text)
        payloads.append({
            "type": "match",
            "year": year,
            "team1": match["team1"],
            "team2": match["team2"],
            "winner": match["winner"],
            "venue": match["venue"],
            "margin": match.get("margin", ""),
            "text": text,
        })
        ids.append(year * 100000 + i)

    # Team season summaries
    for j, team in enumerate(data.get("team_stats", []), 1):
        pos = team.get("position", 10)
        is_qual = team.get('qualified_playoffs', pos <= 4)
        is_champ = team.get('won_title', team.get('team') == data.get('winner'))
        
        text = (
            f"IPL {year} team performance: {team['team']}. "
            f"Finished {pos}th with {team.get('wins', 0)} wins, "
            f"{team.get('losses', 0)} losses, NRR {team.get('nrr', 0.0)}, "
            f"{'qualified for playoffs' if is_qual else 'did not qualify'}. "
            f"{'Won the title!' if is_champ else ''}"
        )
        texts.append(text)
        payloads.append({
            "type": "team_season",
            "year": year,
            "team": team["team"],
            "position": pos,
            "wins": team.get("wins", 0),
            "nrr": team.get("nrr", 0.0),
            "text": text,
        })
        ids.append(year * 100000 + 10000 + j)

    log.info(f"Embedding {len(texts)} chunks for year {year}...")
    vectors = await get_embeddings_batch(texts)

    points = [
        PointStruct(id=pid, vector=vec, payload=pay)
        for pid, vec, pay in zip(ids, vectors, payloads)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    log.info(f"Upserted {len(points)} points for year {year} into Qdrant")


# ─── SQLite ────────────────────────────────────────────────────────────────────
# Note: SQLite uses INTEGER PRIMARY KEY for auto-increment by default
SQL_INIT = """
CREATE TABLE IF NOT EXISTS team_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    matches INTEGER, wins INTEGER, losses INTEGER, no_result INTEGER,
    points INTEGER, nrr FLOAT,
    position INTEGER, qualified_playoffs BOOLEAN, won_title BOOLEAN,
    UNIQUE(year, team)
);

CREATE TABLE IF NOT EXISTS match_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE,
    year INTEGER NOT NULL,
    date TEXT, team1 TEXT, team2 TEXT, winner TEXT,
    margin TEXT, venue TEXT, toss_winner TEXT, toss_decision TEXT,
    team1_score TEXT, team2_score TEXT,
    player_of_match TEXT, match_type TEXT, summary TEXT
);

CREATE TABLE IF NOT EXISTS head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team1 TEXT NOT NULL, team2 TEXT NOT NULL,
    year_from INTEGER, year_to INTEGER,
    total_matches INTEGER, team1_wins INTEGER, team2_wins INTEGER,
    UNIQUE(team1, team2, year_from, year_to)
);
"""


async def ingest_sqlite(db: aiosqlite.Connection, data: dict):
    year = data["year"]

    # Team stats
    for t in data.get("team_stats", []):
        pos = t.get("position", 10)
        is_qual = t.get('qualified_playoffs', pos <= 4)
        is_champ = t.get('won_title', t.get('team') == data.get('winner'))
        
        await db.execute("""
            INSERT INTO team_season_stats
                (year,team,matches,wins,losses,no_result,points,nrr,position,qualified_playoffs,won_title)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (year, team) DO UPDATE SET
                wins=excluded.wins, losses=excluded.losses,
                points=excluded.points, nrr=excluded.nrr,
                position=excluded.position,
                qualified_playoffs=excluded.qualified_playoffs,
                won_title=excluded.won_title
        """, (year, t["team"], t.get("matches", 0), t.get("wins", 0), t.get("losses", 0),
              t.get("no_result", 0), t.get("points", 0), t.get("nrr", 0.0),
              pos, is_qual, is_champ))

    # Matches
    for m in data.get("matches", []):
        await db.execute("""
            INSERT INTO match_results
                (match_id,year,date,team1,team2,winner,margin,venue,
                 toss_winner,toss_decision,team1_score,team2_score,
                 player_of_match,match_type,summary)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (match_id) DO NOTHING
        """, (m["match_id"], year, m["date"], m["team1"], m["team2"],
              m["winner"], m["margin"], m["venue"],
              m.get("toss_winner", ""), m.get("toss_decision", ""),
              m["team1_score"], m["team2_score"],
              m.get("player_of_match", ""), m.get("match_type", "league"), m["summary"]))
    
    await db.commit()
    log.info(f"SQLite: ingested year {year}")


async def compute_h2h(db: aiosqlite.Connection):
    """Compute and store all head-to-head records across all years."""
    async with db.execute("SELECT team1, team2, winner, year FROM match_results") as cursor:
        rows = await cursor.fetchall()
        
    if not rows:
        return

    min_year = min(r[3] for r in rows)
    max_year = max(r[3] for r in rows)

    h2h: dict[tuple, dict] = {}
    for r in rows:
        team1, team2, winner = r[0], r[1], r[2]
        key = tuple(sorted([team1, team2]))
        if key not in h2h:
            h2h[key] = {"total": 0, "wins": {key[0]: 0, key[1]: 0}}
        h2h[key]["total"] += 1
        if winner in h2h[key]["wins"]:
            h2h[key]["wins"][winner] += 1

    for (t1, t2), stats in h2h.items():
        await db.execute("""
            INSERT INTO head_to_head (team1,team2,year_from,year_to,total_matches,team1_wins,team2_wins)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (team1,team2,year_from,year_to) DO UPDATE SET
                total_matches=excluded.total_matches,
                team1_wins=excluded.team1_wins,
                team2_wins=excluded.team2_wins
        """, (t1, t2, min_year, max_year, stats["total"], stats["wins"].get(t1, 0), stats["wins"].get(t2, 0)))

    await db.commit()
    log.info(f"H2H records computed for range {min_year}-{max_year}")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(years: list[int], skip_qdrant: bool = False, skip_postgres: bool = False, shared_qdrant: Optional[QdrantClient] = None):
    qdrant = get_qdrant(shared_qdrant)
    ensure_collection(qdrant)

    db: Optional[aiosqlite.Connection] = None
    if not skip_postgres:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        db = await aiosqlite.connect(SQLITE_DB_PATH)
        await db.executescript(SQL_INIT)

    async with httpx.AsyncClient() as http:
        for year in years:
            path = DATA_DIR / f"ipl_{year}.json"
            if not path.exists():
                log.warning(f"No data file for {year}, run scraper first")
                continue

            data = json.loads(path.read_text())

            if not skip_qdrant:
                await upsert_to_qdrant(qdrant, year, data)

            if not skip_postgres and db:
                await ingest_sqlite(db, data)

    if not skip_postgres and db:
        await compute_h2h(db)
        await db.close()

    log.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL Data Ingestion Pipeline")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2008, 2027)))
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-postgres", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.years, args.skip_qdrant, args.skip_postgres))
