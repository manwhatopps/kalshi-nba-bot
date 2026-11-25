"""Market discovery: every open NBA, NFL and college football game and player market on Kalshi, as buyable sides.

One row per (market, side): what it costs to buy now (the ask), the bid, the size shown, the settlement rule text,
when the game starts, and the keys the rest of the system needs (game, team, player, line). Read-only, no key needed.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.history import NBA_SERIES, NFL_SERIES, START_OFFSET
from bot.props_study import STAT

from . import gamedata as G

NCAAF_SERIES = ["KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "KXNCAAFTEAMTOTAL"]
SERIES = {"NBA": NBA_SERIES, "NFL": NFL_SERIES, "NCAAF": NCAAF_SERIES}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def kind_of(series: str) -> str:
    k = G.kind_of(series)
    if k != "other" and not re.search(r"(1H|1Q|2H|PRECISE)", series):
        return k
    return "prop" if series in STAT else "other"


def scan(k, sports=("NBA", "NFL", "NCAAF"), hours: float = 120, include_live: bool = False) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows, errors = [], []
    for sport in sports:
        for s in SERIES[sport]:
            try:
                ms = k.markets(s)
            except Exception as e:
                errors.append(f"{s}: {str(e)[:80]}")
                continue
            kind = kind_of(s)
            for m in ms:
                occ = m.get("occurrence_datetime") or m.get("expected_expiration_time")
                if not occ or "-" not in m.get("event_ticker", "") or m.get("status") != "active":
                    continue
                start = datetime.fromisoformat(occ.replace("Z", "+00:00")) - START_OFFSET[sport]
                live = start < now + timedelta(minutes=10)
                if (live and not include_live) or start > now + timedelta(hours=hours):
                    continue
                bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
                if not (0 <= bid < ask <= 1):
                    continue
                parts = m["ticker"].split("-")
                code = m["event_ticker"].split("-")[1]
                last = parts[-1]
                team = re.match(r"^([A-Z]+)", last)
                base = dict(ticker=m["ticker"], event_ticker=m["event_ticker"], series=s, sport=sport, kind=kind, game=code,
                            game_key=f"{sport}:{code}", title=m.get("title", ""), rules=(m.get("rules_primary") or "")[:400],
                            start=start, live=live, spread=round(ask - bid, 4), line=_f(m.get("floor_strike")) if m.get("floor_strike") is not None else float("nan"),
                            team_code=(team.group(1) if team and kind != "total" else ""),
                            player=(parts[2] if kind == "prop" and len(parts) >= 3 else ""), scanned=now)
                if ask < 1:
                    rows.append(base | dict(side="yes", ask=ask, bid=bid, mid=(bid + ask) / 2, ask_size=_f(m.get("yes_ask_size_fp"))))
                if bid > 0:
                    rows.append(base | dict(side="no", ask=round(1 - bid, 4), bid=round(1 - ask, 4), mid=1 - (bid + ask) / 2, ask_size=_f(m.get("yes_bid_size_fp"))))
    d = pd.DataFrame(rows)
    d.attrs["errors"] = errors
    return d
