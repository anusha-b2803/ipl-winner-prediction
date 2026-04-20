"""
IPL RAG Prediction API
FastAPI backend serving predictions, stats, and Q&A endpoints.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import datetime

# Make sure sibling packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global sync state
_sync_status = {
    "last_sync": "Never",
    "is_syncing": False,
    "total_syncs": 0
}

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ipl_stats.db")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/vector_db")

# Cache predictions in memory to avoid repeated LLM calls.
# Fix #9: Live-season (current year) entries expire after LIVE_CACHE_TTL_HOURS.
# Historical seasons are cached indefinitely (data never changes).
_prediction_cache: dict[int, dict] = {}
_prediction_cache_ts: dict[int, datetime.datetime] = {}
LIVE_CACHE_TTL_HOURS = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    log.info(f"Using SQLite database at {SQLITE_DB_PATH}")
    
    # ─── Environment Validation ───
    required_keys = ["GOOGLE_API_KEY", "HUGGINGFACE_API_KEY", "SCRAPER_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        log.warning(f"⚠️ Missing recommended API keys: {', '.join(missing)}")
        log.warning("Some features (LLM, ScraperAPI) may be disabled or limited.")

    # ─── Bootstrap & Sync ───
    import asyncio
    from scripts.bootstrap_render import bootstrap
    from rag.predictor import get_embed_model, load_transformer_weights
    
    # Pre-warm AI models during startup to prevent timeout on first request
    log.info("Pre-warming AI models...")
    get_embed_model()
    load_transformer_weights()
    
    # Run full bootstrap check in background
    asyncio.create_task(bootstrap())
    
    # Periodic background sync (12h)
    asyncio.create_task(run_periodic_sync())
    
    yield
    # Safely close Qdrant connection to avoid file locks on Uvicorn reload
    try:
        from rag.predictor import _qdrant_client
        if _qdrant_client is not None:
            _qdrant_client.close()
            log.info("Closed QdrantClient cleanly.")
    except Exception as e:
        log.warning(f"Error closing Qdrant: {e}")

async def run_periodic_sync():
    """Background task to sync data every 12 hours."""
    from pipeline.sync_pipeline import run_sync
    import asyncio
    
    # Wait a bit after startup
    await asyncio.sleep(30)
    
    while True:
        try:
            _sync_status["is_syncing"] = True
            success = await run_sync()
            if success:
                _sync_status["last_sync"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _sync_status["total_syncs"] += 1
            _sync_status["is_syncing"] = False
        except Exception as e:
            log.error(f"Background sync error: {e}")
            _sync_status["is_syncing"] = False
            
        # Wait 12 hours
        await asyncio.sleep(12 * 3600)


app = FastAPI(
    title="IPL Winner Prediction API",
    description="RAG-powered IPL winner prediction using historical statistics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint for health checks and service confirmation."""
    return {
        "status": "online",
        "service": "IPL Winner Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/stats/sync-status")
async def get_sync_status():
    """Get the current automated sync status."""
    return _sync_status


# ─── Pydantic Models ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    year: int
    predicted_winner: str
    confidence: float
    reasoning: str
    top_teams: list[dict]
    key_factors: list[str]
    historical_context: str
    retrieved_chunks: int
    cached: bool = False
    momentum: Optional[dict[str, list]] = None # Added: {team: [W, L, ...]}


class QueryRequest(BaseModel):
    question: str
    year: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    year: Optional[int]


class TeamStatsResponse(BaseModel):
    year: int
    teams: list[dict]


class H2HResponse(BaseModel):
    team1: str
    team2: str
    total_matches: int
    team1_wins: int
    team2_wins: int
    team1_win_pct: float
    team2_win_pct: float


# ─── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = os.path.exists(SQLITE_DB_PATH)
    google_ok = os.getenv("GOOGLE_API_KEY") is not None
    return {
        "status": "ok",
        "database": db_ok,
        "ai_provider": "google",
        "google_api_configured": google_ok,
        "sync_status": _sync_status
    }


# ─── Prediction Endpoints ───────────────────────────────────────────────────────

@app.get("/predict/{year}", response_model=PredictionResponse)
async def predict_winner(year: int, force_refresh: bool = Query(default=False)):
    """Predict the IPL winner for a given year using RAG."""
    if year < 2008 or year > 2030:
        raise HTTPException(status_code=400, detail="Year must be between 2008 and 2030")

    # Serve from cache if available.
    # Fix #9: Expire cache for the current live season after TTL.
    current_year = datetime.datetime.now().year
    if not force_refresh and year in _prediction_cache:
        ts = _prediction_cache_ts.get(year)
        is_live_season = (year >= current_year)
        cache_expired = (
            is_live_season
            and ts is not None
            and (datetime.datetime.now() - ts).total_seconds() > LIVE_CACHE_TTL_HOURS * 3600
        )
        if not cache_expired:
            cached = _prediction_cache[year]
            return PredictionResponse(**cached, cached=True)
        elif cache_expired:
            log.info(f"Cache expired for live season {year}, fetching fresh prediction.")
            del _prediction_cache[year]
            del _prediction_cache_ts[year]

    try:
        from rag.predictor import predict_ipl_winner
        result = await predict_ipl_winner(year)
        data = {
            "year": year,
            "predicted_winner": result.predicted_winner,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "top_teams": result.top_teams,
            "key_factors": result.key_factors,
            "historical_context": result.historical_context,
            "retrieved_chunks": result.retrieved_chunks,
        }
        _prediction_cache[year] = data
        _prediction_cache_ts[year] = datetime.datetime.now()
        return PredictionResponse(**data, cached=False)
    except Exception as e:
        log.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/latest/", response_model=PredictionResponse)
async def predict_latest():
    """Predict the most recent IPL season's winner."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        async with db.execute("SELECT MAX(year) FROM team_season_stats") as cursor:
            row = await cursor.fetchone()
            year = row[0] if row else None
            
    if not year:
        raise HTTPException(status_code=404, detail="No season data available")
    return await predict_winner(int(year))


# ─── Stats Endpoints ────────────────────────────────────────────────────────────

@app.get("/stats/seasons", response_model=list[dict])
async def list_seasons():
    """List all available seasons."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        async with db.execute("""
            SELECT year, 
                   MAX(CASE WHEN won_title = 1 THEN team END) as winner,
                   COUNT(DISTINCT team) as teams
            FROM team_season_stats
            GROUP BY year ORDER BY year DESC
        """) as cursor:
            rows = await cursor.fetchall()
            return [{"year": r[0], "winner": r[1], "teams": r[2]} for r in rows]


@app.get("/stats/{year}", response_model=TeamStatsResponse)
async def get_season_stats(year: int):
    """Get full standings for a given season, fetching live for current years."""
    # Live fetch for current/ongoing seasons
    current_year = datetime.datetime.now().year
    if year >= current_year:
        try:
            from scripts.live_update import update_current_season
            # Run the update (synchronously here for stats fetch, 
            # or could be backgrounded, but user wants stats NOW)
            await update_current_season()
        except Exception as e:
            log.error(f"Live data fetch failed: {e}")

    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        async with db.execute("SELECT * FROM team_season_stats WHERE year=? ORDER BY position", (year,)) as cursor:
            rows = await cursor.fetchall()
            teams = []
            team_names = [r[2] for r in rows]
            
            # Pull momentum if current year
            momentum = {}
            if year == current_year:
                from rag.predictor import get_recent_matches_for_teams
                momentum = await get_recent_matches_for_teams(db, team_names, year=year)

            for r in rows:
                teams.append({
                    "team": r[2],
                    "matches": r[3],
                    "wins": r[4],
                    "losses": r[5],
                    "points": r[7],
                    "nrr": r[8],
                    "position": r[9],
                    "qualified_playoffs": bool(r[10]),
                    "won_title": bool(r[11]),
                    "momentum": momentum.get(r[2], [])
                })
    return {"year": year, "teams": teams}


@app.get("/stats/team/{team_name}")
async def get_team_history(team_name: str):
    """Get a team's full history across all seasons."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM team_season_stats WHERE team LIKE ? ORDER BY year",
            (f"%{team_name}%",)
        ) as cursor:
            rows = await cursor.fetchall()
        
        async with db.execute(
            "SELECT COUNT(*) FROM team_season_stats WHERE team LIKE ? AND won_title=1",
            (f"%{team_name}%",)
        ) as cursor:
            row = await cursor.fetchone()
            title_count = row[0] if row else 0

    if not rows:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    
    return {
        "team": rows[0]["team"],
        "titles": title_count,
        "seasons": [dict(r) for r in rows],
    }


@app.get("/stats/h2h/{team1}/{team2}", response_model=H2HResponse)
async def get_head_to_head(team1: str, team2: str):
    """Get head-to-head record between two teams."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM head_to_head
            WHERE (team1 LIKE ? AND team2 LIKE ?)
               OR (team1 LIKE ? AND team2 LIKE ?)
            LIMIT 1
        """, (f"%{team1}%", f"%{team2}%", f"%{team2}%", f"%{team1}%")) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="H2H record not found")

    total = row["total_matches"] or 1
    return H2HResponse(
        team1=row["team1"], team2=row["team2"],
        total_matches=row["total_matches"],
        team1_wins=row["team1_wins"], team2_wins=row["team2_wins"],
        team1_win_pct=round(row["team1_wins"] / total, 3),
        team2_win_pct=round(row["team2_wins"] / total, 3),
    )


@app.get("/stats/titles/all")
async def all_title_counts():
    """Get total IPL title counts per team."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT team, COUNT(*) as titles,
                   group_concat(year) as years
            FROM team_season_stats
            WHERE won_title = 1
            GROUP BY team ORDER BY titles DESC
        """) as cursor:
            rows = await cursor.fetchall()
    
    return [
        {"team": r["team"], "titles": r["titles"], "years": [int(y) for y in (r["years"].split(",") if r["years"] else [])]}
        for r in rows
    ]


@app.get("/stats/matches/{year}")
async def get_matches(year: int, limit: int = Query(default=50, le=200)):
    """Get match results for a season."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM match_results WHERE year=? ORDER BY date LIMIT ?",
            (year, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


# ─── Q&A Endpoint ───────────────────────────────────────────────────────────────

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    """Answer a free-form IPL question using RAG."""
    try:
        from rag.predictor import answer_cricket_query
        answer = await answer_cricket_query(req.question, req.year)
        return QueryResponse(question=req.question, answer=answer, year=req.year)
    except Exception as e:
        log.error(f"Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Pipeline trigger (admin) ───────────────────────────────────────────────────

@app.post("/admin/run-pipeline")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    years: str = Query(default="2024", description="Comma-separated years"),
    secret: str = Query(default=""),
):
    """Trigger scrape + ingest pipeline (requires ADMIN_SECRET env var)."""
    admin_secret = os.getenv("ADMIN_SECRET")
    if not admin_secret:
        log.warning("ADMIN_SECRET not set, blocking admin access")
        raise HTTPException(status_code=403, detail="Admin access disabled (no secret configured)")
        
    if secret != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    year_list = [int(y.strip()) for y in years.split(",")]

    async def _run():
        import subprocess
        for y in year_list:
            subprocess.run(["python", "scraper/ipl_scraper.py", "--years", str(y)])
            subprocess.run(["python", "pipeline/ingest.py", "--years", str(y)])
        log.info(f"Pipeline complete for years: {year_list}")

    background_tasks.add_task(_run)
    return {"message": f"Pipeline started for years: {year_list}"}


@app.post("/admin/update-live")
async def trigger_live_update(background_tasks: BackgroundTasks, secret: str = Query(default="")):
    """Manually trigger a live standings refresh for the current year."""
    admin_secret = os.getenv("ADMIN_SECRET", "admin")
    if secret != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    from scripts.live_update import update_current_season
    background_tasks.add_task(update_current_season)
    return {"message": f"Live update task queued for IPL {datetime.datetime.now().year}"}


if __name__ == "__main__":
    import uvicorn
    # Check if we should update requirements and env
    print(f"\n[API] Starting IPL Prophet API (Zero-Infra Mode)")
    print(f"[*] DB: {SQLITE_DB_PATH}")
    print(f"[*] Vectors: {QDRANT_PATH}")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
