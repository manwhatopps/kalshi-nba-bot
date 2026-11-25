"""The fund's record: one SQLite file (data/quantsport.db) for runs, decisions, cards, paper positions, approvals and
equity, plus one compressed snapshot per run of every side that was scanned (data/qs_scans/). Timestamps are UTC ISO."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from bot import config

DB = config.DATA_DIR / "quantsport.db"
SCANS = config.DATA_DIR / "qs_scans"
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, ts TEXT, mode TEXT, model_version TEXT, scanned INT, games INT, accepted INT, notes TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, sport TEXT, kind TEXT, ticker TEXT, side TEXT,
  game_key TEXT, title TEXT, tier TEXT, decision TEXT, p_est REAL, p_se REAL, p_market REAL, p_model REAL, ask REAL, fee REAL, ev REAL, ev_low REAL,
  kelly REAL, stake REAL, contracts REAL, wait_price REAL, source TEXT, thesis TEXT, audit TEXT, model_version TEXT,
  result TEXT DEFAULT '', pnl REAL, close_price REAL);
CREATE TABLE IF NOT EXISTS cards(card_id TEXT PRIMARY KEY, run_id TEXT, ts TEXT, tier TEXT, target REAL, stake REAL, p_joint REAL, fair_mult REAL,
  need_mult REAL, ev_at_need REAL, n_legs INT, legs TEXT, games TEXT, expires TEXT, status TEXT, audit TEXT, result TEXT DEFAULT '', pnl REAL, legs_hit INT);
CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY AUTOINCREMENT, opened TEXT, mode TEXT, kind TEXT, ref TEXT, tier TEXT, sport TEXT, label TEXT,
  stake REAL, pays REAL, tickers TEXT, games TEXT, teams TEXT, players TEXT, expires TEXT, status TEXT DEFAULT 'open', closed TEXT, pnl REAL);
CREATE TABLE IF NOT EXISTS approvals(token TEXT PRIMARY KEY, created TEXT, expires TEXT, ticker TEXT, side TEXT, limit_price REAL, quantity REAL,
  max_cost REAL, used TEXT, sig TEXT);
CREATE TABLE IF NOT EXISTS equity(ts TEXT PRIMARY KEY, bankroll REAL, open_exposure REAL, realized REAL, peak REAL, drawdown REAL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or DB))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def insert(con, table: str, row: dict) -> int:
    row = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()}
    cur = con.execute(f"INSERT OR REPLACE INTO {table}({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
    con.commit()
    return cur.lastrowid


def frame(con, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def save_scan(run_id: str, d: pd.DataFrame) -> None:
    """Every side scanned in this run with our estimate at the time: the raw material for calibration later."""
    SCANS.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("ticker", "side", "sport", "kind", "game_key", "ask", "bid", "mid", "p_est", "p_se", "p_model", "ev", "ev_low", "source", "start") if c in d]
    x = d[cols].assign(run_id=run_id)
    x.attrs = {}                                   # attrs hold live objects that do not belong in the file
    x.to_parquet(SCANS / f"{run_id}.parquet", index=False)
