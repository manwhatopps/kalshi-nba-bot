"""Live paper-trader for every rule in the fund's registry (no orders are placed).

Polls Kalshi once a minute during games and logs a paper entry, at the real quote, the first time
a rule's condition is met in a market:

  ml95 / ml97   NBA moneyline: either side can be bought at >= 95c / 97c
  dd_under      NBA double-double: the UNDER (NO) can be bought at 50-80c

Only the fund's primary graded edge gets a dollar stake; everything else is tracked at $0 so each
rule keeps building an out-of-sample record. The displayed size at the quote is recorded too, which
is the evidence on how much could really be filled.  Log: data/ledger/live_rules.csv
"""
from __future__ import annotations

import csv
import time
from datetime import datetime, timezone

from . import config
from .history import START_OFFSET
from .kalshi import data_client

LOG = config.DATA_DIR / "ledger" / "live_rules.csv"
FIELDS = ["ts", "rule", "ticker", "side", "price", "size_shown", "mins_into_game", "title", "stake", "result", "pnl", "unit"]
RULES = [dict(id="ml95_nba", series="KXNBAGAME", kind="touch", th=0.95),
         dict(id="ml97_nba", series="KXNBAGAME", kind="touch", th=0.97),
         dict(id="dd_under", series="KXNBA2D", kind="band", side="no", lo=0.50, hi=0.80)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def fund_stake(k) -> tuple[str, float]:
    """(rule id, dollars per bet) for the fund's primary graded edge; ('', 0) when nothing is graded A or B."""
    try:
        from . import fund
        from .report_fund import primary_edge
        cfg = fund.load_cfg()
        e = primary_edge(fund.edge_registry(cfg["z"]))
        if e is None:
            return "", 0.0
        core = cfg["core_pct"] * cfg["bankroll"]
        return e.id, round(min(fund.stake_fraction(e) * cfg["bankroll"], core / max(e.per_night, 1)), 2)
    except Exception as ex:
        print(f"! fund stake unavailable: {ex}", flush=True)
        return "", 0.0


def _seen() -> set:
    if not LOG.exists() or LOG.stat().st_size == 0:
        return set()
    with open(LOG) as f:
        return {(r["rule"], r["ticker"], r["side"]) for r in csv.DictReader(f)}


def run(hours: float = 8, every: int = 60, league: str = "NBA"):
    k = data_client()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    paid_rule, stake = fund_stake(k)
    print(f"fund plan: ${stake} per bet on '{paid_rule or 'nothing (core in cash)'}'; other rules tracked at $0", flush=True)
    seen, end = _seen(), time.time() + hours * 3600
    staked_games = set()
    new = not LOG.exists() or LOG.stat().st_size == 0
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
            f.flush()
        while time.time() < end:
            now = datetime.now(timezone.utc)
            for series in sorted({r["series"] for r in RULES}):
                try:
                    ms = k.markets(series)
                except Exception as e:
                    print(f"{now:%H:%M} ! {series}: {e}", flush=True)
                    continue
                for m in ms:
                    occ = m.get("occurrence_datetime")
                    if not occ or "-" not in m.get("event_ticker", ""):
                        continue
                    mins = (now - (datetime.fromisoformat(occ.replace("Z", "+00:00")) - START_OFFSET[league])).total_seconds() / 60
                    if not (0 <= mins <= 170):
                        continue
                    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
                    quotes = {"yes": (ask, _f(m.get("yes_ask_size_fp"))),
                              "no": ((1 - bid) if bid > 0 else 0.0, _f(m.get("yes_bid_size_fp")))}
                    game = m["event_ticker"].split("-")[1]
                    for r in (x for x in RULES if x["series"] == series):
                        for side, (price, size) in quotes.items():
                            if r["kind"] == "touch":
                                hit = r["th"] <= price <= 0.99
                            else:
                                hit = side == r["side"] and r["lo"] <= price < r["hi"]
                            key = (r["id"], m["ticker"], side)
                            if not hit or key in seen:
                                continue
                            seen.add(key)
                            unit = (r["id"], game) if r["kind"] == "touch" else key   # a moneyline game is one bet
                            st = stake if (r["id"] == paid_rule and unit not in staked_games) else 0.0
                            staked_games.add(unit)
                            w.writerow(dict(ts=now.isoformat(timespec="seconds"), rule=r["id"], ticker=m["ticker"], side=side,
                                            price=round(price, 2), size_shown=size, mins_into_game=round(mins),
                                            title=m.get("title", ""), stake=st, result="", pnl="", unit=""))
                            f.flush()
                            print(f"{now:%H:%M} PAPER {r['id']} {side.upper()} {m['ticker']} at {price:.2f} "
                                  f"(size shown {size:.0f}, stake ${st})", flush=True)
            time.sleep(every)


def settle(k) -> None:
    import pandas as pd
    if not LOG.exists() or LOG.stat().st_size == 0:
        return
    df = pd.read_csv(LOG, dtype={"result": str}).fillna({"result": ""})
    if df.empty:
        return
    todo = sorted(set(df.ticker[df.result == ""]))
    res = {}
    for i in range(0, len(todo), 100):
        try:
            d = k.request("GET", "/markets", params={"tickers": ",".join(todo[i:i + 100]), "limit": 100})
            res.update({m["ticker"]: m.get("result") or "" for m in d.get("markets") or []})
        except Exception as e:
            print(f"  ! settle live rules: {e}")
    for i in df.index[df.result == ""]:
        r = res.get(df.at[i, "ticker"], "")
        if r in ("yes", "no"):
            win = r == df.at[i, "side"]
            p = float(df.at[i, "price"])
            c = p + 0.07 * p * (1 - p)
            st = float(df.at[i, "stake"]) if pd.notna(df.at[i, "stake"]) else 0.0
            df.at[i, "result"] = "won" if win else "lost"
            df.at[i, "pnl"] = round(st * (1 / c - 1) if win else -st, 4)
            df.at[i, "unit"] = round((1 - c) if win else -c, 4)
    df.to_csv(LOG, index=False)
    done = df[df.result.isin(["won", "lost"])]
    if len(done):
        print("\nLIVE PAPER RECORD by rule (real quotes, graded):")
        for rule, g in done.groupby("rule"):
            unit = pd.to_numeric(g.unit, errors="coerce").fillna(0)
            cost = g.price + 0.07 * g.price * (1 - g.price)
            print(f"  {rule:10s} {len(g):4d} entries, won {(g.result == 'won').mean():.1%}, avg price {g.price.mean():.3f}, "
                  f"return {unit.sum() / cost.sum():+.1%}, median size shown {g.size_shown.median():.0f} contracts")
        money = done[pd.to_numeric(done.stake, errors="coerce").fillna(0) > 0]
        if len(money):
            from . import fund
            b = fund.load_cfg()["bankroll"]
            pnl = pd.to_numeric(money.pnl).sum()
            print(f"  PAPER FUND core sleeve: {len(money)} bets, {int((money.result == 'won').sum())} won, "
                  f"P&L ${pnl:+.2f} -> paper bankroll ${b + pnl:,.2f} (started ${b:,.2f})")
