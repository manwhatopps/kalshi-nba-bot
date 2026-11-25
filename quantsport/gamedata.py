"""Game-level data for the three sports, and Kalshi's own pre-game prices for game markets.

  results(sport)      one row per finished game: date, home, away, points, neutral site (free sources, no key)
  kalshi_pre()        one row per settled Kalshi game market (winner / spread / total / team total): the last price
                      traded in the three hours before the start, and how it settled

Sources: NBA team totals from the player game logs already used by the bot (Hugging Face cbratkovics/nba-game-logs,
CC0); NFL from nflverse `schedules/games` (CC-BY 4.0); college football from sportsdataverse/cfbfastR-data schedule
files, which carry CollegeFootballData results without needing an API key. Kalshi NBA/NFL prices come from the raw
fill files (two seasons); college football from Kalshi's candlestick history (this season).
"""
from __future__ import annotations

import io
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

from bot import config

EXT = config.DATA_DIR / "ext"
PRE = config.DATA_DIR / "game_pre.parquet"
GAME_SERIES = {"NBA": ["KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "KXNBATEAMTOTAL"],
               "NFL": ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL"],
               "NCAAF": ["KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "KXNCAAFTEAMTOTAL"]}
KIND = {"GAME": "winner", "SPREAD": "spread", "TOTAL": "total", "TEAMTOTAL": "team_total"}
CFB = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/parquet/cfb_schedules_{y}.parquet"
NFL_GAMES = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
NFL_ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "ARZ": "ARI"}   # Kalshi -> nflverse


def kind_of(series: str) -> str:
    for suffix, k in (("TEAMTOTAL", "team_total"), ("SPREAD", "spread"), ("TOTAL", "total"), ("GAME", "winner")):
        if series.endswith(suffix):
            return k
    return "other"


def _fetch(url: str, name: str, max_age_h: float = 12) -> pd.DataFrame:
    import time
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / name
    if not f.exists() or (time.time() - f.stat().st_mtime) / 3600 > max_age_h:
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        except Exception as e:
            if not f.exists():
                raise RuntimeError(f"{name} unavailable: {e}")
    return pd.read_parquet(f)


def results(sport: str) -> pd.DataFrame:
    """Finished games, oldest first: date, season, home, away, home_pts, away_pts, neutral."""
    if sport == "NBA":
        from bot.data import load_logs
        L = load_logs(topup=False)
        g = L.groupby(["game_id", "game_date", "season", "team", "home"], as_index=False).pts.sum()
        h, a = g[g.home == 1], g[g.home == 0]
        d = h.merge(a, on=["game_id", "game_date", "season"], suffixes=("_h", "_a"))
        d = pd.DataFrame({"date": d.game_date, "season": d.season, "home": d.team_h, "away": d.team_a,
                          "home_pts": d.pts_h.astype(float), "away_pts": d.pts_a.astype(float), "neutral": False})
        d["home"], d["away"] = d.home.replace({"SAN": "SAS"}), d.away.replace({"SAN": "SAS"})
    elif sport == "NFL":
        g = _fetch(NFL_GAMES, "nfl/games_live.parquet", 6)
        g = g[(g.season >= 2022) & g.home_score.notna()]
        d = pd.DataFrame({"date": pd.to_datetime(g.gameday), "season": g.season.astype(str), "home": g.home_team, "away": g.away_team,
                          "home_pts": g.home_score.astype(float), "away_pts": g.away_score.astype(float),
                          "neutral": (g.location == "Neutral")})
    elif sport == "NCAAF":
        parts = []
        for y in (2023, 2024, 2025, 2026):
            try:
                parts.append(_fetch(CFB.format(y=y), f"cfb_schedules_{y}.parquet", 6 if y == 2026 else 9999))
            except Exception as e:
                print(f"  ! college football {y}: {e}")
        g = pd.concat(parts, ignore_index=True)
        g = g[g.home_points.notna() & g.away_points.notna()]
        start = pd.to_datetime(g.start_date, utc=True)
        d = pd.DataFrame({"date": start.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(), "season": g.season.astype(str),
                          "home": g.home_team, "away": g.away_team, "home_pts": g.home_points.astype(float),
                          "away_pts": g.away_points.astype(float), "neutral": g.neutral_site.fillna(False).astype(bool),
                          "home_fbs": (g.home_division == "fbs"), "away_fbs": (g.away_division == "fbs")})
    else:
        raise ValueError(sport)
    return d.sort_values("date").reset_index(drop=True)


def upcoming(sport: str) -> pd.DataFrame:
    """Scheduled games not yet played (NFL and college football; NBA pairings come from Kalshi's own events)."""
    if sport == "NFL":
        g = _fetch(NFL_GAMES, "nfl/games_live.parquet", 6)
        g = g[g.home_score.isna()]
        return pd.DataFrame({"date": pd.to_datetime(g.gameday), "home": g.home_team, "away": g.away_team, "neutral": (g.location == "Neutral")})
    if sport == "NCAAF":
        g = _fetch(CFB.format(y=2026), "cfb_schedules_2026.parquet", 6)
        g = g[g.home_points.isna()]
        start = pd.to_datetime(g.start_date, utc=True)
        return pd.DataFrame({"date": start.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(), "home": g.home_team,
                             "away": g.away_team, "neutral": g.neutral_site.fillna(False).astype(bool)})
    return pd.DataFrame(columns=["date", "home", "away", "neutral"])


# ---------------------------------------------------------------- names
def _norm(name: str) -> str:
    n = (name or "").lower().replace("&", "and")
    n = re.sub(r"\bst\.?(?=\s|$)", "state", n)
    n = n.replace("(fl)", "").replace("miami (oh)", "miami ohio").replace("(oh)", "ohio")
    return re.sub(r"[^a-z0-9 ]", "", n).strip()


def match_name(name: str, options: list[str]) -> tuple[str | None, float]:
    """Best match for a Kalshi team name among schedule names (exact after normalising, else closest spelling)."""
    n = _norm(name)
    best, score = None, 0.0
    for o in options:
        m = _norm(o)
        s = 1.0 if m == n else SequenceMatcher(None, n, m).ratio() - (0.15 if (n in m or m in n) and n != m and abs(len(n) - len(m)) > 6 else 0)
        if s > score:
            best, score = o, s
    return best, score


# ---------------------------------------------------------------- Kalshi pre-game prices for game markets
def build_pre(rebuild: bool = False) -> pd.DataFrame:
    """Last traded YES price in the 3 hours before the start, for every settled NBA/NFL game-level market on file
    (needs the raw fill files, like the other summaries), plus college football from the candle history."""
    from bot.trades import PRIOR_DIR, PRIOR_RESULTS, RESULTS, TRADES_DIR
    if PRE.exists() and not rebuild:
        return pd.read_parquet(PRE)
    series = GAME_SERIES["NBA"] + GAME_SERIES["NFL"]
    out = []
    for res_f, tdir in ((PRIOR_RESULTS, PRIOR_DIR), (RESULTS, TRADES_DIR)):
        if not res_f.exists() or not tdir.exists():
            continue
        r = pd.read_parquet(res_f)
        r = r[r.series.isin(series) & r.result.isin(["yes", "no"])][["ticker", "series", "league", "game", "title", "floor_strike", "result", "tipoff"]]
        tip = r.set_index("ticker").tipoff
        best = {}
        import functools
        import gc
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        flt = functools.reduce(lambda a, b: a | b, [pc.starts_with(pc.field("ticker"), s_ + "-") for s_ in series])
        for f in sorted(tdir.glob("2*.parquet")):          # one day at a time, game-level tickers only, to stay inside memory
            t = ds.dataset(f).to_table(columns=["ticker", "yes_price", "created_time"], filter=flt).to_pandas()
            if t.empty:
                continue
            t = t[t.ticker.isin(tip.index)]
            mins = (t.created_time - t.ticker.map(tip)).dt.total_seconds() / 60
            t = t[(mins >= -180) & (mins < 0)]
            if len(t):
                last = t.loc[t.groupby("ticker").created_time.idxmax()]
                for tk, pr, ct in zip(last.ticker, last.yes_price, last.created_time):
                    if tk not in best or ct > best[tk][1]:
                        best[tk] = (float(pr), ct)
            del t
            gc.collect()
        best = pd.DataFrame({"ticker": list(best), "pre": [v[0] for v in best.values()]}) if best else None
        if best is not None:
            out.append(r.merge(best, on="ticker"))
    h = config.DATA_DIR / "history_v2"
    for s in GAME_SERIES["NCAAF"]:
        f = h / f"{s}.parquet"
        if f.exists():
            c = pd.read_parquet(f)
            c = c[c.result.isin(["yes", "no"]) & c.pre_bid.notna() & c.pre_ask.notna()]
            c = c.assign(pre=(c.pre_bid + c.pre_ask) / 2, league="NCAAF")
            out.append(c[["ticker", "series", "league", "game", "title", "floor_strike", "result", "tipoff", "pre"]])
    d = pd.concat(out, ignore_index=True)
    d["kind"] = d.series.map(kind_of)
    d["pre"] = d.pre.astype(float)
    d["y"] = (d.result == "yes").astype(float)
    d.to_parquet(PRE, index=False)
    return d
