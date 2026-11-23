"""NBA player game logs.

Primary source: the CC0 Hugging Face dataset `cbratkovics/nba-game-logs` (refreshed nightly,
one row per player-game). Optional top-up: `nba_api` (stats.nba.com) for games newer than the
dataset — works from a home connection, usually blocked from cloud servers.
"""
from __future__ import annotations

import time
import unicodedata
import re

import pandas as pd
import requests

from . import config

KALSHI_TO_LOG_TEAM = {"SAS": "SAN"}  # dataset uses SAN for San Antonio
LOG_TO_KALSHI_TEAM = {v: k for k, v in KALSHI_TO_LOG_TEAM.items()}
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _download(season: str, max_age_h: float) -> pd.DataFrame:
    path = config.DATA_DIR / f"game_logs_{season}.parquet"
    stale = not path.exists() or (time.time() - path.stat().st_mtime) / 3600 > max_age_h
    if stale:
        url = (f"https://huggingface.co/datasets/{config.HF_DATASET}/resolve/main/"
               f"game_logs/game_logs_{season}.parquet")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        except Exception as e:  # keep using the cached copy if refresh fails
            if not path.exists():
                raise RuntimeError(f"could not download {url}: {e}") from e
    return pd.read_parquet(path)


def _nba_api_topup(after: pd.Timestamp) -> pd.DataFrame | None:
    try:
        from nba_api.stats.endpoints import leaguegamelog
        y = int(config.CURRENT_SEASON[:4])
        df = leaguegamelog.LeagueGameLog(season=f"{y}-{str(y + 1)[2:]}", player_or_team_abbreviation="P",
                                         date_from_nullable=(after + pd.Timedelta(days=1)).strftime("%m/%d/%Y"),
                                         timeout=20).get_data_frames()[0]
    except Exception:
        return None
    if df.empty:
        return None
    out = pd.DataFrame({
        "game_id": df.GAME_ID, "game_date": pd.to_datetime(df.GAME_DATE), "season": config.CURRENT_SEASON,
        "player_id": df.PLAYER_ID, "player_name": df.PLAYER_NAME, "team": df.TEAM_ABBREVIATION,
        "opponent": df.MATCHUP.str[-3:], "home": ~df.MATCHUP.str.contains("@"), "minutes": df.MIN.astype(float),
        "pts": df.PTS, "reb": df.REB, "ast": df.AST, "fg3m": df.FG3M, "stl": df.STL, "blk": df.BLK,
        "source": "nba_api"})
    out["team"] = out.team.replace({"SAS": "SAN"})
    out["opponent"] = out.opponent.replace({"SAS": "SAN"})
    return out[out.minutes > 0]


def load_logs(max_age_h: float = 12, topup: bool = True) -> pd.DataFrame:
    frames = [_download(s, max_age_h) for s in config.SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df.game_date)
    if topup:
        extra = _nba_api_topup(df.game_date.max())
        if extra is not None:
            df = pd.concat([df, extra[~extra.game_id.isin(df.game_id)]], ignore_index=True)
    df = df[df.minutes > 0].copy()
    df["name_key"] = df.player_name.map(norm_name)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
