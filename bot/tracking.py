"""Shot-zone and tracking profile of what every defense ALLOWS, game by game (no lookahead).

Source: github.com/gabriel1200/player_sheets (daily team-game tracking tables built from NBA.com
tracking and pbpstats; the repo has no licence file, so the files are downloaded at run time for
private analysis and never committed or redistributed). `all_games_2026` = the 2025-26 season,
`all_games_2027` = 2026-27.

For each defense and game we take the OPPONENT's offensive row from the same game, then average the
defense's previous 15 games. That shows what a scheme concedes far better than a box score:

  rim / mid-range / corner-three / above-the-break-three share of opponent shots
  catch-and-shoot threes and pull-up jumpers allowed, wide-open threes allowed
  drives, paint touches, post touches and potential assists allowed, seconds per touch (isolation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from . import config

EXT = config.DATA_DIR / "ext"
URL = "https://raw.githubusercontent.com/gabriel1200/player_sheets/master/teamgame_report/year_files/all_games_{y}.parquet"
ALLOWED = ["a_rim", "a_mid", "a_corner3", "a_three", "a_cs3", "a_pullup", "a_wide_open3", "a_drives", "a_paint_touch",
           "a_post_touch", "a_pot_ast", "a_sec_touch", "a_pts_paint"]


def load(years=(2026, 2027), max_age_h: float = 12) -> pd.DataFrame:
    import time
    EXT.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years:
        f = EXT / f"all_games_{y}.parquet"
        if not f.exists() or (time.time() - f.stat().st_mtime) / 3600 > max_age_h:
            try:
                r = requests.get(URL.format(y=y), timeout=120)
                r.raise_for_status()
                f.write_bytes(r.content)
            except Exception as e:
                if not f.exists():
                    print(f"  ! tracking data for {y} unavailable: {e}")
                    continue
        parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True).copy()
    extra = pd.DataFrame({"team": d.TEAM_ABBREVIATION.replace({"SAS": "SAN"}),
                          "game_date": pd.to_datetime(d["date"].astype(str), format="%Y%m%d")}, index=d.index)
    return pd.concat([d, extra], axis=1)


def allowed(d: pd.DataFrame | None = None, window: int = 15) -> pd.DataFrame:
    """Per defense and game: averages over its PREVIOUS `window` games of what opponents did against it."""
    d = d if d is not None else load()
    if d.empty:
        return d
    fga = d.FGA.clip(lower=1)
    o = pd.DataFrame({
        "GameId": d.GameId, "game_date": d.game_date, "offense": d.team,
        "a_rim": d.RA_FGA / fga, "a_mid": d.MID_FGA / fga, "a_corner3": d.CORNER_3_FGA / fga, "a_three": d.FG3A / fga,
        "a_cs3": d.CATCH_SHOOT_FG3A / fga, "a_pullup": d.PULL_UP_FGA / fga, "a_wide_open3": d.wide_open_FG3A_FREQUENCY,
        "a_drives": 100 * d.DRIVES / d.team_poss.clip(lower=50), "a_paint_touch": 100 * d.PAINT_TOUCHES / d.team_poss.clip(lower=50),
        "a_post_touch": 100 * d.POST_TOUCHES / d.team_poss.clip(lower=50), "a_pot_ast": d.POTENTIAL_AST / fga,
        "a_sec_touch": d.AVG_SEC_PER_TOUCH, "a_pts_paint": 100 * d.PTS_PAINT / d.team_poss.clip(lower=50)})
    pairs = d[["GameId", "team"]].merge(d[["GameId", "team"]], on="GameId", suffixes=("", "_o"))
    pairs = pairs[pairs.team != pairs.team_o].rename(columns={"team": "defense", "team_o": "offense"})
    t = pairs.merge(o, on=["GameId", "offense"]).sort_values(["defense", "game_date"])
    g = t.groupby("defense")
    out = t[["defense", "game_date", "GameId"]].copy()
    for c in ALLOWED:
        out[c] = g[c].transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
    return out


SCHEMES = [  # (label, {metric: direction}) - every listed metric must be at least 0.6 sd in that direction
    ("drop-style: gives up mid-range and pull-ups, takes away threes", {"a_mid": 1, "a_pullup": 1, "a_three": -1}),
    ("packs the paint and concedes threes", {"a_rim": -1, "a_three": 1}),
    ("zone-like: catch-and-shoot threes and ball movement allowed", {"a_cs3": 1, "a_pot_ast": 1, "a_three": 1}),
    ("switch-style: forces isolation, few assisted looks", {"a_pot_ast": -1, "a_sec_touch": 1}),
    ("gives up the rim", {"a_rim": 1, "a_pts_paint": 1}),
    ("runs shooters off the line", {"a_three": -1, "a_wide_open3": -1}),
    ("leaves shooters open", {"a_wide_open3": 1}),
    ("keeps drivers out of the lane", {"a_drives": -1, "a_paint_touch": -1}),
]


def latest_schemes(a: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest allowed-profile per defense this season, with z-scores and scheme labels."""
    a = a if a is not None else allowed()
    if a.empty:
        return a
    cur = a[a.game_date >= a.game_date.max() - pd.Timedelta(days=45)].dropna(subset=["a_rim"])
    s = cur.sort_values("game_date").groupby("defense").tail(1).set_index("defense")
    z = (s[ALLOWED] - s[ALLOWED].mean()) / s[ALLOWED].std()
    labels = []
    for team in s.index:
        lab = [name for name, cond in SCHEMES if all(z.at[team, m] * sgn >= 0.6 for m, sgn in cond.items())]
        labels.append("; ".join(lab))
    s = s.assign(scheme=labels)
    for c in ("a_rim", "a_mid", "a_three"):
        s[c + "_rank"] = s[c].rank(ascending=False).astype(int)     # 1 = allows the most
    s["iso_third"] = pd.qcut(s.a_sec_touch, 3, labels=["low", "mid", "high"]).astype(str)   # high = forces isolation
    return s
