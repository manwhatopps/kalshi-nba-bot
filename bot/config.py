"""All settings come from environment variables (or a .env file). See .env.example."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BOT_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

KALSHI_ENV = os.getenv("KALSHI_ENV", "demo").lower()  # "demo" until you deliberately switch to "prod"
_BASES = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}
# Market data (scanning) always reads PROD prices — demo markets are fake. Orders go to KALSHI_ENV.
DATA_BASE_URL = os.getenv("KALSHI_DATA_BASE_URL", _BASES["prod"])
TRADE_BASE_URL = os.getenv("KALSHI_BASE_URL", _BASES.get(KALSHI_ENV, _BASES["demo"]))

KALSHI_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# --- risk limits (dollars). Parlays lose most of the time; these caps are the point. ---
BANKROLL = float(os.getenv("BANKROLL", "100"))
MAX_STAKE_PER_COMBO = float(os.getenv("MAX_STAKE_PER_COMBO", "2"))
MAX_DAILY_STAKE = float(os.getenv("MAX_DAILY_STAKE", "10"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.1"))

# --- model blending: how much to trust our stats model vs. the market price ---
W_MODEL = float(os.getenv("W_MODEL", "0.4"))   # weight of stats model
W_BOOK = float(os.getenv("W_BOOK", "0.4"))     # weight of de-vigged sportsbook (only if ODDS_API_KEY)
# remainder goes to the Kalshi mid price

HF_DATASET = os.getenv("HF_DATASET", "cbratkovics/nba-game-logs")
SEASONS = os.getenv("SEASONS", "2024-25,2025-26,2026-27").split(",")
CURRENT_SEASON = SEASONS[-1]
