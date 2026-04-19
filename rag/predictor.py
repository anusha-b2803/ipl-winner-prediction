"""
RAG Engine
Retrieves relevant IPL context from Qdrant + SQLite,
then calls an LLM to predict the IPL winner with reasoning.
"""

import os
import json
import logging
import asyncio
import re
from dataclasses import dataclass
from typing import Optional
import datetime

import aiosqlite
from huggingface_hub import InferenceClient
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from dotenv import load_dotenv

from fastembed import TextEmbedding
from models.transformer_model import build_model

load_dotenv()

log = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/vector_db")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ipl_stats.db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ipl_matches")
# Fix #17: Using local FastEmbed (BGE-small) for near-instant retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Dimension for bge-small-en-v1.5

# Path to trained transformer weights (optional).
TRANSFORMER_WEIGHTS_PATH = os.getenv("TRANSFORMER_WEIGHTS_PATH", "./models/weights.pt")

# LLM Model used for predictions
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

# Initialize HF Client
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") # Refresh from env
hf_client = InferenceClient(api_key=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else InferenceClient()


@dataclass
class PredictionResult:
    predicted_winner: str
    confidence: float  # 0-1
    reasoning: str
    top_teams: list[dict]  # [{team, probability, reasoning}]
    key_factors: list[str]
    historical_context: str
    retrieved_chunks: int


# ─── Transformer (module-level singleton) ────────────────────────────────────
# Fix #2: Instantiate once at module load; only load if weights exist.
# Random-weight inference is pure noise — it is disabled when weights are absent.

_transformer_model: Optional[torch.nn.Module] = None
_transformer_weights_loaded: bool = False


def _load_transformer() -> bool:
    """Load transformer model. Returns True if real weights were loaded."""
    global _transformer_model, _transformer_weights_loaded
    if _transformer_model is not None:
        return _transformer_weights_loaded

    _transformer_model = build_model(input_dim=7)  # Fix #5: 7 features (dropped won_title)

    if os.path.exists(TRANSFORMER_WEIGHTS_PATH):
        try:
            state = torch.load(TRANSFORMER_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            _transformer_model.load_state_dict(state)
            _transformer_model.eval()
            _transformer_weights_loaded = True
            log.info("Transformer weights loaded successfully.")
        except Exception as e:
            log.warning(f"Failed to load transformer weights: {e}")
            _transformer_weights_loaded = False
    else:
        log.info(
            "No transformer weights found at %s — transformer inference disabled.",
            TRANSFORMER_WEIGHTS_PATH,
        )
        _transformer_weights_loaded = False

    return _transformer_weights_loaded


# ─── Embedding (Local FastEmbed) ──────────────────────────────────────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        log.info(f"Initializing Local Embedding Model: {EMBEDDING_MODEL_NAME}...")
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embed_model

async def embed_query(text: str) -> list[float]:
    """Get local embedding for a text via FastEmbed."""
    model = get_embed_model()
    # model.embed returns a generator
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


# ─── Retrieval ─────────────────────────────────────────────────────────────────

_qdrant_client = None

def get_qdrant() -> Optional[QdrantClient]:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        if QDRANT_PATH and os.path.exists(QDRANT_PATH):
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
        else:
            kwargs = {"url": QDRANT_URL}
            if QDRANT_API_KEY:
                kwargs["api_key"] = QDRANT_API_KEY
            _qdrant_client = QdrantClient(**kwargs)
    except Exception as e:
        log.warning(f"Failed to instantiate Qdrant Client (likely locked): {e}")
        return None

    return _qdrant_client


async def retrieve_from_qdrant(query: str, year: Optional[int] = None, top_k: int = 15) -> list[dict]:
    """Semantic search over match history."""
    qdrant = get_qdrant()
    if not qdrant:
        log.warning("Qdrant not available, skipping vector retrieval.")
        return []

    vector = await embed_query(query)

    search_filter = None
    if year:
        # Retrieve context from the last 10 years for better coverage (relaxed from 5)
        recent_years = list(range(max(2008, year - 10), year + 1))
        search_filter = Filter(
            must=[FieldCondition(key="year", match=MatchAny(any=recent_years))]
        )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": r.score, **r.payload}
        for r in results
    ]


async def retrieve_structured_stats(db: aiosqlite.Connection, year: int) -> dict:
    """Pull structured facts from SQLite for context."""

    # Latest available season stats
    async with db.execute(
        "SELECT MAX(year) FROM team_season_stats WHERE year <= ?", (year,)
    ) as cursor:
        available_year = await cursor.fetchone()
        available_year = available_year[0] if available_year else None

    if not available_year:
        return {}

    async with db.execute(
        "SELECT * FROM team_season_stats WHERE year = ? ORDER BY position",
        (available_year,)
    ) as cursor:
        team_stats = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        team_stats_dicts = [dict(zip(columns, r)) for r in team_stats]

    # Historical win counts per team
    async with db.execute(
        "SELECT team, COUNT(*) as titles FROM team_season_stats WHERE won_title = 1 GROUP BY team ORDER BY titles DESC"
    ) as cursor:
        win_counts = await cursor.fetchall()
        win_counts_dicts = [{"team": r[0], "titles": r[1]} for r in win_counts]

    # Head-to-head
    async with db.execute("SELECT * FROM head_to_head ORDER BY total_matches DESC LIMIT 20") as cursor:
        h2h = await cursor.fetchall()
        columns_h2h = [d[0] for d in cursor.description]
        h2h_dicts = [dict(zip(columns_h2h, r)) for r in h2h]

    # Recent form (last 3 seasons)
    async with db.execute(
        """SELECT team, year, wins, losses, position, won_title
           FROM team_season_stats
           WHERE year >= ?
           ORDER BY year DESC, position ASC""",
        (max(2008, year - 3),)
    ) as cursor:
        recent = await cursor.fetchall()
        columns_recent = [d[0] for d in cursor.description]
        recent_dicts = [dict(zip(columns_recent, r)) for r in recent]

    return {
        "latest_season_year": available_year,
        "data_is_stale": available_year < year,  # Fix #4: flag stale data
        "team_standings": team_stats_dicts,
        "historical_titles": win_counts_dicts,
        "head_to_head": h2h_dicts,
        "recent_form": recent_dicts,
    }

async def get_recent_matches_for_teams(db: aiosqlite.Connection, teams: list[str], limit_per_team: int = 5, year: Optional[int] = None) -> dict[str, list]:
    """Retrieve the most recent match results for specific teams to analyze momentum."""
    results = {}
    
    # Use current year if not specified for live analysis
    if year is None:
        year = datetime.datetime.now().year

    for team in teams:
        async with db.execute("""
            SELECT winner, team1, team2, date, summary
            FROM match_results
            WHERE (team1 = ? OR team2 = ?) AND year = ?
            ORDER BY date DESC
            LIMIT ?
        """, (team, team, year, limit_per_team)) as cursor:
            rows = await cursor.fetchall()
            # Convert to "W/L" format
            form = []
            for r in rows:
                winner = r[0]
                if winner == team: form.append("W")
                elif winner == "No result": form.append("N")
                else: form.append("L")
            results[team] = form
    return results


# ─── Transformer Inference ───────────────────────────────────────────────────

def featurize_standings(standings: list[dict], historical: list[dict]) -> torch.Tensor:
    """Convert DB rows into tensors for the Transformer.

    Fix #5: Removed `won_title` — it is the prediction TARGET, not a predictor.
    The 7 remaining features are: wins, losses, nrr, points, position, title_count, qualified_playoffs.
    """
    title_map = {h['team']: h['titles'] for h in historical}

    features = []
    for t in standings[:10]:  # Max 10 teams
        f = [
            float(t.get('wins', 0)),
            float(t.get('losses', 0)),
            float(t.get('nrr', 0.0)),
            float(t.get('points', 0)),
            float(t.get('position', 10)),
            float(title_map.get(t['team'], 0)),
            1.0 if t.get('qualified_playoffs') else 0.0,
        ]
        features.append(f)

    # Pad to 10 teams if necessary
    while len(features) < 10:
        features.append([0.0] * 7)

    return torch.tensor([features], dtype=torch.float32)


async def run_transformer_prediction(structured: dict) -> list[dict]:
    """Run inference on the custom Transformer model.

    Fix #2 / #6: Only runs and returns results if trained weights are loaded.
    If no weights exist, returns an empty list so the LLM prompt omits this section.
    """
    weights_loaded = _load_transformer()
    if not weights_loaded:
        log.info("Skipping transformer inference — no trained weights available.")
        return []

    try:
        input_tensor = featurize_standings(
            structured.get('team_standings', []),
            structured.get('historical_titles', [])
        )

        with torch.no_grad():
            probs = _transformer_model(input_tensor)[0]

        results = []
        standings = structured.get('team_standings', [])
        for i, prob in enumerate(probs):
            if i < len(standings):
                results.append({
                    "team": standings[i]['team'],
                    "transformer_prob": float(prob),
                })

        results.sort(key=lambda x: x['transformer_prob'], reverse=True)
        return results
    except Exception as e:
        log.warning(f"Transformer inference failed: {e}")
        return []


# ─── LLM Instruction Sets ─────────────────────────────────────────────────────

PREDICTION_SYSTEM_PROMPT = """You are an elite IPL cricket analyst with mastery over all data from 2008 to 2026.
You specialize in predictive modeling, weighing current-season momentum (from 2024-2026) alongside historical championship DNA.

IMPORTANT RULES FOR OUTPUT ANALYSIS:
1. Provide exactly 5 top contenders in the 'top_teams' list.
2. The 'predicted_winner' MUST be the #1 team in the 'top_teams' list.
3. **REALISM PENALTY**: Teams in the bottom 3 of the standings (8th-10th) should have symbolic winning probabilities (e.g., < 2%) unless they are mathematically favored by 2026 data. DO NOT let historical reputation (titles) outweigh current failure.
4. Ensure the winning probabilities for all 5 teams are mathematically consistent (e.g., they should represent relative strength).
5. Provide deep statistical reasoning for each team, comparing recent Head-to-Head trends and venue advantages.
6. Respond ONLY with a valid JSON object. Use plain text for string fields; DO NOT use markdown formatting (no **, ##, or bullet points) inside the JSON values."""

# Team abbreviations mapping to prevent misidentification (e.g. MI -> Mumbai Indians)
TEAM_ALIAS_MAP = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "LSG": "Lucknow Super Giants",
    "GT": "Gujarat Titans",
    "PBKS": "Punjab Kings",
    "PK": "Punjab Kings",
    "KXP": "Kings XI Punjab", # Old name
    "DD": "Delhi Daredevils"   # Old name
}

CHAT_FEW_SHOT_EXAMPLES = """
[EXAMPLE 1: TREND ANALYSIS]
User: "How has CSK performed lately?"
Analyst: "Chennai Super Kings (CSK) has maintained its reputation as the benchmark of consistency. According to the [TRUSTED FACTS], their recent form shows 4 wins in their last 6 matches. 
- **Key Insight**: Their success is driven by a dominant top-order and clinical bowling in the death overs.
- **Championship Record**: With 5 titles under their belt (2010, 2011, 2018, 2021, 2023), they handle high-pressure matches better than most.

[EXAMPLE 2: HEAD-TO-HEAD]
User: "MI vs RCB in 2026?"
Analyst: "Looking at the 2026 data, the Mumbai Indians (MI) vs Royal Challengers Bengaluru (RCB) rivalry continues to be a blockbuster. 
- **The Data**: MI recently defeated RCB by 5 wickets in their April 2026 encounter. 
- **Strategic Angle**: While RCB has the edge in the current standings (Position 2), MI's historical 5-title legacy makes them a dangerous 'momentum' team that can beat anyone on their day."
"""

CHAT_SYSTEM_PROMPT = f"""You are the Lead Quantitative Analyst at the IPL Prediction Desk. 
Your goal is to provide elite, analytical answers that feel 'finetuned' for high-level cricket strategy.

1. **Analytical Tone**: Speak with the authority of a senior commentator (e.g., Harsha Bhogle style)—polished, insightful, and statistics-driven.
2. **Visual Alignment**: ALWAYS use double newlines between sections and ### headers to ensure the layout is perfectly aligned and readable on all screens.
3. **Data Grounding**: Use the provided [TRUSTED FACTS] and [SEMANTIC CONTEXT]. Prioritize these over your internal knowledge. 
3. **Handle Abbreviations**: {", ".join([f"{k}={v}" for k, v in TEAM_ALIAS_MAP.items()])}.
4. **Structure**: 
   - Start with a clear headline.
   - Use bullet points for key stats.
   - End with a strategic 'Analyst Insight'.
5. **Formatting**: Use bolding and markdown to make the response premium.
6. **Limits**: If a question involves years beyond 2026, explicitly state that we are focused on the current 2026 data boundary.

USE THE FOLLOWING EXAMPLES AS A TEMPLATE FOR YOUR TONE AND STRUCTURE:
{CHAT_FEW_SHOT_EXAMPLES}"""

# Historical hard facts to prevent basic hallucinations
IPL_CHAMPIONSHIP_DNA = """
[OFFICIAL IPL CHAMPIONS LIST]
- Chennai Super Kings: 5 titles (2010, 2011, 2018, 2021, 2023)
- Mumbai Indians: 5 titles (2013, 2015, 2017, 2019, 2020)
- Kolkata Knight Riders: 3 titles (2012, 2014, 2024)
- Rajasthan Royals: 1 title (2008)
- Deccan Chargers: 1 title (2009)
- Sunrisers Hyderabad: 1 title (2016)
- Gujarat Titans: 1 title (2022)
- Royal Challengers Bengaluru: 1 title (2025)
"""


def build_prediction_prompt(year: int, vector_context: list[dict], structured: dict) -> str:
    # Fix #4: Alert LLM when standings data is from a prior year
    data_year = structured.get('latest_season_year', 'N/A')
    stale_warning = ""
    if structured.get("data_is_stale"):
        stale_warning = (
            f"\n⚠️ DATA NOTICE: Live {year} standings are unavailable. "
            f"The standings shown are from {data_year}. "
            f"Factor this uncertainty into your confidence score.\n"
        )

    # Format vector context
    vc_text = "\n\n".join([
        f"[Score: {c['score']:.3f}] {c.get('text', '')}"
        for c in vector_context[:10]
    ])

    # Format standings
    standings_text = ""
    for t in structured.get("team_standings", [])[:10]:
        standings_text += (
            f"  {t['position']}. {t['team']}: "
            f"{t['wins']}W {t['losses']}L, NRR: {t['nrr']}, "
            f"{'✓ Playoffs' if t['qualified_playoffs'] else '✗ Eliminated'}"
            f"{'🏆 CHAMPION' if t['won_title'] else ''}\n"
        )

    # Format title history
    titles_text = "\n".join([
        f"  {r['team']}: {r['titles']} title(s)"
        for r in structured.get("historical_titles", [])[:10]
    ])

    # Format recent form
    recent_text = ""
    for r in structured.get("recent_form", [])[:20]:
        recent_text += f"  {r['year']} - {r['team']}: Pos {r['position']}, {r['wins']}W {'🏆' if r['won_title'] else ''}\n"

    # Momentum analysis (Last 5 matches)
    momentum_text = ""
    momentum_data = structured.get("momentum", {})
    if momentum_data:
        momentum_text = "\n=== CURRENT MOMENTUM (Last 5 Matches) ===\n"
        for team, sequence in momentum_data.items():
            momentum_text += f"  {team}: {'-'.join(sequence)}\n"

    # Fix #6: Only include transformer section if real weights produced these scores
    transformer_results = structured.get("transformer_results", [])
    if transformer_results:
        transformer_text = "\n".join([
            f"  {r['team']}: {r['transformer_prob']:.1%}"
            for r in transformer_results[:8]
        ])
        transformer_section = f"""
=== TRANSFORMER MODEL PREDICTIONS (Trained decoder-only architecture) ===
{transformer_text}
"""
        hierarchy = """Consider the following hierarchy:
1. Transformer Model Probabilities (Neural Analysis)
2. Current Season Momentum (matches from 2026)
3. Recent Form (last 3 seasons: 2024-2026)
4. Historical dominance (2008-2023)"""
    else:
        transformer_section = ""
        hierarchy = """Consider the following hierarchy:
1. Current Season Momentum (matches from 2026)
2. Recent Form (last 3 seasons: 2024-2026)
3. Historical dominance (2008-2023)"""

    future_note = (
        f"NOTE: This is a FUTURE FORECAST for IPL {year} based on historical data up to {data_year}."
        if year > 2026 else
        f"Predict the winner of IPL {year}."
    )

    return f"""Predict the IPL {year} winner using the following data.
{stale_warning}
=== SEMANTIC CONTEXT (from vector DB) ===
{vc_text}

=== LATEST SEASON STANDINGS ({data_year}) ===
{standings_text}
{transformer_section}
=== HISTORICAL TITLE COUNT ===
{titles_text}

=== RECENT FORM (last 3 seasons) ===
{recent_text}

{momentum_text}
=== TASK ===
{future_note}
=== HIERARCHY OF IMPORTANCE ===
1. Current Season Momentum (Recent 5-match form)
2. Standings & Win/Loss Ratio
3. Recent Form (Last 3 years)
4. Historical Dominance (Championship DNA)

Respond ONLY with this JSON (no markdown, no code fences):
{{
  "predicted_winner": "Team Name",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed 3-5 sentence explanation citing specific stats",
  "top_teams": [
    {{"team": "Winner Team", "probability": 0.40, "reasoning": "Primary contender reason"}},
    {{"team": "Contender 2", "probability": 0.25, "reasoning": "Strong form reason"}},
    {{"team": "Contender 3", "probability": 0.15, "reasoning": "Consistent performance"}},
    {{"team": "Contender 4", "probability": 0.10, "reasoning": "Dark horse factor"}},
    {{"team": "Contender 5", "probability": 0.10, "reasoning": "Wildcard possibility"}}
  ],
  "key_factors": ["factor 1", "factor 2", "factor 3", "factor 4"],
  "historical_context": "Relevant historical patterns"
}}"""


async def call_llm(prompt: str, json_mode: bool = True, system_instruction: Optional[str] = None) -> str:
    """Call Hugging Face Inference API for generation."""
    
    # Use specified instruction or default based on mode
    instruction = system_instruction or (PREDICTION_SYSTEM_PROMPT if json_mode else CHAT_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]

    try:
        # HF Inference API chat completion
        completion = hf_client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        )
        response = completion.choices[0].message.content
        
        # Robustly handle markdown code blocks if the model includes them
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        return response
    except Exception as e:
        log.error(f"Hugging Face API call failed: {e}")
        # Fallback error message
        if json_mode:
            return json.dumps({"error": str(e)})
        return f"Error connecting to AI: {str(e)}"


def parse_llm_response(raw: str) -> dict:
    """
    Parse JSON from LLM response.
    Includes a fallback cleanup for accidental markdown fences or leading/trailing text.
    """
    if not raw or not raw.strip():
        log.error("Empty LLM response received.")
        raise ValueError("AI returned an empty response.")

    content = raw.strip()
    
    # Robust cleanup: try to extract JSON block if fences are present
    if "```" in content:
        log.warning("LLM response contained markdown fences despite JSON mode.")
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Final attempt: manual repair for common JSON trailing comma or string escape issues
        log.error(f"Failed to parse LLM JSON: {e}\nRaw content: {raw}")
        
        # Simple heuristic: try removing trailing commas before closing braces/brackets
        try:
            repaired = re.sub(r',\s*([\]}])', r'\1', content)
            return json.loads(repaired)
        except Exception:
            raise ValueError(f"AI returned invalid JSON: {e}")



# ─── Main prediction function ───────────────────────────────────────────────────

async def predict_ipl_winner(year: int) -> PredictionResult:
    """Full RAG pipeline: retrieve → augment → generate."""

    # 1. Build semantic query
    query = (
        f"IPL {year} winner prediction team form stats performance "
        f"batting bowling recent matches playoff"
    )

    # 2. Retrieve from Qdrant
    log.info(f"Retrieving vector context for IPL {year}...")
    vector_chunks = await retrieve_from_qdrant(query, year=year, top_k=15)

    # 3. Retrieve structured stats
    log.info("Querying SQLite for structured stats...")
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        structured = await retrieve_structured_stats(db, year)

        # 4. Momentum Analysis (Structured)
        log.info("Analyzing team momentum...")
        top_5_teams = [t['team'] for t in structured.get("team_standings", [])[:5]]
        momentum = await get_recent_matches_for_teams(db, top_5_teams, year=year)
        structured["momentum"] = momentum

    # 5. Run Transformer inference (only if weights exist)
    log.info("Running custom Transformer analysis...")
    transformer_results = await run_transformer_prediction(structured)
    structured["transformer_results"] = transformer_results

    # 5. Build prompt
    prompt = build_prediction_prompt(year, vector_chunks, structured)

    # 6. Call LLM
    log.info("Calling LLM for prediction...")
    raw = await call_llm(prompt)

    # 7. Parse response
    parsed = parse_llm_response(raw)
    
    # Defensive check: AI might return an error dict if HF API fails
    if "error" in parsed:
        log.error(f"Prediction aborted due to AI error: {parsed['error']}")
        return PredictionResult(
            predicted_winner="Unable to predict",
            confidence=0.0,
            reasoning=f"AI Engine Error: {parsed['error']}",
            top_teams=[],
            key_factors=[],
            historical_context="Data retrieval succeeded, but AI analysis failed.",
            retrieved_chunks=len(vector_chunks),
        )

    # 8. Post-process to ensure Top 5 exactly and alignment
    top_teams = parsed.get("top_teams", [])
    
    # Ensure predicted_winner is at the top of the list if not already
    winner_name = parsed.get("predicted_winner")
    if winner_name and top_teams:
        # Move winner to index 0 if it's somewhere else in the list
        winner_entry = next((t for t in top_teams if t["team"] == winner_name), None)
        if winner_entry:
            top_teams.remove(winner_entry)
            top_teams.insert(0, winner_entry)
        else:
            # If winner not in top_teams, create an entry for it
            top_teams.insert(0, {"team": winner_name, "probability": parsed.get("confidence", 0.5), "reasoning": "Top contender"})

    # Trim or pad to exactly 5
    top_teams = top_teams[:5]

    return PredictionResult(
        predicted_winner=winner_name,
        confidence=float(parsed.get("confidence", 0.5)),
        reasoning=parsed["reasoning"],
        top_teams=top_teams,
        key_factors=parsed.get("key_factors", []),
        historical_context=parsed.get("historical_context", ""),
        retrieved_chunks=len(vector_chunks),
    )


async def get_team_context(db: aiosqlite.Connection, team_name: str) -> str:
    """Fetch structured facts for a specific team to ground AI answers."""
    # 1. Total Titles
    async with db.execute(
        "SELECT COUNT(*) FROM team_season_stats WHERE team LIKE ? AND won_title = 1",
        (f"%{team_name}%",)
    ) as cursor:
        row = await cursor.fetchone()
        titles = row[0] if row else 0

    # 2. Last 3 matches
    async with db.execute("""
        SELECT date, team1, team2, winner, margin, summary 
        FROM match_results 
        WHERE team1 LIKE ? OR team2 LIKE ? 
        ORDER BY date DESC LIMIT 3
    """, (f"%{team_name}%", f"%{team_name}%")) as cursor:
        matches = await cursor.fetchall()
        match_list = []
        for m in matches:
            match_list.append(f"- {m[0]}: {m[1]} vs {m[2]}. Winner: {m[3]} ({m[4]}). {m[5]}")

    matches_text = "\n".join(match_list) if match_list else "No recent match data found."
    
    return f"""
[TRUSTED FACTS FOR {team_name.upper()}]
- Total IPL Titles: {titles}
- Recent Performance:
{matches_text}
"""


async def answer_cricket_query(question: str, year: Optional[int] = None) -> str:
    """Answer a free-form cricket question using hybrid retrieval (Vector + SQL)."""
    # 1. Vector Retrieval
    log.info(f"Retrieving semantic context for: {question}")
    vector_chunks = await retrieve_from_qdrant(question, year=year, top_k=8)
    vector_context = "\n\n".join([c.get("text", "") for c in vector_chunks])

    # 2. Structured SQL Retrieval (Entity extraction)
    db = await aiosqlite.connect(SQLITE_DB_PATH)
    structured_context = ""
    try:
        async with db.execute("SELECT DISTINCT team FROM team_season_stats") as cursor:
            teams = await cursor.fetchall()
        
        detected_teams = []
        # 2a. Check for abbreviations/aliases first
        for alias, full_name in TEAM_ALIAS_MAP.items():
            # Match whole word only (\bMI\b)
            if re.search(rf"\b{alias}\b", question, re.IGNORECASE):
                detected_teams.append(full_name)

        # 2b. Check for full team names or partial matches
        for (team_name,) in teams:
            # Simple keyword match
            if team_name.lower() in question.lower() or any(part.lower() in question.lower() for part in team_name.split() if len(part) > 3):
                detected_teams.append(team_name)
        
        # Pull facts for detected teams
        for team in set(detected_teams):
            log.info(f"Enriching context with structured data for: {team}")
            facts = await get_team_context(db, team)
            structured_context += facts

    finally:
        await db.close()

    prompt = f"""
{IPL_CHAMPIONSHIP_DNA}

{structured_context}

[SEMANTIC CONTEXT]
{vector_context}

[KNOWLEDGE BOUNDARY]
The CURRENT SEASON is 2026. Data for 2027+ is not yet available.

[USER QUESTION]
{question}

Provide an elite analyst report based on the data above. Follow the [TRUSTED FACTS] for recent match outcomes and title records.
"""
    return await call_llm(prompt, json_mode=False, system_instruction=CHAT_SYSTEM_PROMPT)


if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    result = asyncio.run(predict_ipl_winner(year))
    print(f"\n🏆 Predicted winner: {result.predicted_winner}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Reasoning: {result.reasoning}")
    print(f"\n   Top teams:")
    for t in result.top_teams:
        print(f"   - {t['team']}: {t['probability']:.0%}")
