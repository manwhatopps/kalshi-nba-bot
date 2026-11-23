"""Optional: de-vigged sportsbook probabilities from The Odds API (needs ODDS_API_KEY).

Two-sided markets (Over/Under at the same line) are de-vigged with penaltyblog's power method,
then averaged across books. Costs roughly (#markets) credits per game per run — the free tier is
500 credits/month, so keep ODDS_MARKETS short or upgrade.
"""
from __future__ import annotations

import os
from collections import defaultdict

import requests

from . import config
from .data import norm_name

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
STAT_KEY = {("pts",): "player_points", ("reb",): "player_rebounds", ("ast",): "player_assists",
            ("fg3m",): "player_threes", ("pts", "reb", "ast"): "player_points_rebounds_assists",
            ("pts", "reb"): "player_points_rebounds", ("pts", "ast"): "player_points_assists",
            ("reb", "ast"): "player_rebounds_assists", ("blk",): "player_blocks", ("stl",): "player_steals"}
DEFAULT_MARKETS = "h2h,spreads,totals,player_points,player_rebounds,player_assists,player_threes"
TEAMS = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
         "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
         "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
         "IND": "Indiana Pacers", "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers",
         "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
         "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
         "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers",
         "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
         "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"}


def _devig(over: float, under: float) -> float | None:
    try:
        import penaltyblog as pb
        return float(pb.implied.calculate_implied([over, under], method="power").probabilities[0])
    except Exception:
        a, b = 1 / over, 1 / under
        return a / (a + b)


def fetch_book_probs(games: set[tuple[str, str]]) -> dict:
    """games: {(away_code, home_code)}. Returns {key: prob} where key is
    ('prop', name_key, market, line) | ('win', team_code) | ('spread', team_code, line) | ('total', home, line)."""
    if not config.ODDS_API_KEY:
        return {}
    markets = os.getenv("ODDS_MARKETS", DEFAULT_MARKETS)
    ev = requests.get(f"{BASE}/events", params={"apiKey": config.ODDS_API_KEY}, timeout=30).json()
    name_to_code = {v: k for k, v in TEAMS.items()}
    acc = defaultdict(list)
    for e in ev:
        away, home = name_to_code.get(e["away_team"]), name_to_code.get(e["home_team"])
        if (away, home) not in games:
            continue
        r = requests.get(f"{BASE}/events/{e['id']}/odds", timeout=30, params={
            "apiKey": config.ODDS_API_KEY, "regions": "us", "markets": markets, "oddsFormat": "decimal"})
        if r.status_code != 200:
            print(f"  ! odds api {r.status_code}: {r.text[:200]}")
            continue
        for bk in r.json().get("bookmakers", []):
            for mk in bk.get("markets", []):
                key = mk["key"]
                outs = mk.get("outcomes", [])
                if key == "h2h" and len(outs) == 2:
                    p = _devig(outs[0]["price"], outs[1]["price"])
                    acc[("win", name_to_code.get(outs[0]["name"]))].append(p)
                    acc[("win", name_to_code.get(outs[1]["name"]))].append(1 - p)
                    continue
                pairs = defaultdict(dict)
                for o in outs:
                    if key == "spreads":
                        pairs[(o["name"], o.get("point"))]["over"] = o["price"]
                    else:
                        pairs[(o.get("description", ""), o.get("point"))][o["name"].lower()] = o["price"]
                if key == "spreads":  # team -x covers: pair with the other side +x
                    for (team, pt), d in pairs.items():
                        other = [v for (t2, p2), v in pairs.items() if t2 != team and p2 == -pt]
                        if other and pt is not None and pt < 0:
                            p = _devig(d["over"], other[0]["over"])
                            acc[("spread", name_to_code.get(team), -pt)].append(p)
                    continue
                for (desc, pt), d in pairs.items():
                    if "over" not in d or "under" not in d or pt is None:
                        continue
                    p = _devig(d["over"], d["under"])
                    if key == "totals":
                        acc[("total", home, pt)].append(p)
                    else:
                        acc[("prop", norm_name(desc), key, pt)].append(p)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def attach(legs, book: dict) -> int:
    n = 0
    for l in legs:
        if l.kind == "prop":
            k = ("prop", norm_name(l.player), STAT_KEY.get(l.stats), l.line)
        elif l.kind == "win":
            k = ("win", l.team)
        elif l.kind == "spread":
            k = ("spread", l.team, l.line)
        else:
            k = ("total", l.home, l.line)
        if k in book:
            l.p_book = book[k]
            n += 1
    return n
