"""Live paper-trader for the one rule that survived the trade-level backtest:

    NBA moneyline, during the game: the first time a team can be bought at >= 95c, buy it once.

Polls Kalshi once a minute, logs each paper entry at the real ask to data/ledger/live_touch.csv,
and `settle` grades it. No orders are placed.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta, timezone

from . import config
from .history import START_OFFSET
from .kalshi import data_client

LOG = config.DATA_DIR / "ledger" / "live_touch.csv"
FIELDS = ["ts", "ticker", "side", "price", "threshold", "mins_into_game", "title", "stake", "result", "pnl"]


def fund_stake(k, league="NBA") -> tuple[float, float]:
    """(threshold, dollars per trigger) from the fund plan: only the primary graded edge gets money."""
    try:
        from . import fund
        from .report_fund import primary_edge
        from .favorites import live_legs
        cfg = fund.load_cfg()
        e = primary_edge(fund.edge_registry(cfg["z"]))
        if e is None:
            return 0.0, 0.0
        legs = live_legs(k, league, hours_ahead=14, max_spread=0.2, include_live=True)
        games = legs[legs.series == "KXNBAGAME"].game.nunique() if len(legs) else 0
        core = cfg["core_pct"] * cfg["bankroll"]
        return float(e.threshold), round(min(fund.stake_fraction(e) * cfg["bankroll"], core / max(e.game_share * games, 1)), 2)
    except Exception as ex:
        print(f"! fund stake unavailable: {ex}", flush=True)
        return 0.0, 0.0


def _seen() -> set:
    if not LOG.exists() or LOG.stat().st_size == 0:
        return set()
    with open(LOG) as f:
        return {(r["ticker"], r["side"], r["threshold"]) for r in csv.DictReader(f)}


def run(series="KXNBAGAME", thresholds=(0.95, 0.97), hours: float = 8, every: int = 60, league="NBA"):
    k = data_client()
    paid_th, stake = fund_stake(k, league)
    print(f"fund plan: ${stake} per trigger at {paid_th:.2f}; other thresholds are tracked with $0", flush=True)
    seen_games = set()   # the two tickers of a game are one bet: money goes on the first only
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen, end = _seen(), time.time() + hours * 3600
    new = not LOG.exists() or LOG.stat().st_size == 0
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
            f.flush()
        while time.time() < end:
            now = datetime.now(timezone.utc)
            try:
                ms = k.markets(series)
            except Exception as e:
                print(f"{now:%H:%M} ! {e}", flush=True)
                time.sleep(every)
                continue
            for m in ms:
                occ = m.get("occurrence_datetime")
                if not occ:
                    continue
                tip = datetime.fromisoformat(occ.replace("Z", "+00:00")) - START_OFFSET[league]
                mins = (now - tip).total_seconds() / 60
                if not (0 <= mins <= 170):
                    continue
                try:
                    bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
                except ValueError:
                    continue
                for side, price in (("yes", ask), ("no", 1 - bid if bid > 0 else 0)):
                    for th in thresholds:
                        key = (m["ticker"], side, str(th))
                        if th <= price <= 0.99 and key not in seen:
                            seen.add(key)
                            game = m["event_ticker"].split("-")[1]
                            st = stake if (abs(th - paid_th) < 1e-9 and (game, th) not in seen_games) else 0.0
                            seen_games.add((game, th))
                            w.writerow(dict(ts=now.isoformat(timespec="seconds"), ticker=m["ticker"], side=side,
                                            price=round(price, 2), threshold=th, mins_into_game=round(mins),
                                            title=m.get("title", ""), stake=st, result="", pnl=""))
                            f.flush()
                            print(f"{now:%H:%M} PAPER BUY {side.upper()} {m['ticker']} at {price:.2f} "
                                  f"({mins:.0f} min into game)", flush=True)
            time.sleep(every)


def settle(k) -> None:
    import pandas as pd
    if not LOG.exists() or LOG.stat().st_size == 0:
        return
    df = pd.read_csv(LOG, dtype={"result": str}).fillna({"result": ""})
    if df.empty:
        return
    for i in df.index[df.result == ""]:
        try:
            r = k.market(df.at[i, "ticker"]).get("result") or ""
        except Exception:
            r = ""
        if r in ("yes", "no"):
            win = r == df.at[i, "side"]
            p = df.at[i, "price"]
            df.at[i, "result"] = "won" if win else "lost"
            c = p + 0.07 * p * (1 - p)                       # cost per contract with fee
            st = float(df.at[i, "stake"]) if "stake" in df.columns and pd.notna(df.at[i, "stake"]) else 0.0
            df.at[i, "pnl"] = round(st * (1 / c - 1) if win else -st, 4)   # dollars on the fund's stake
            df.at[i, "unit"] = round((1 - c) if win else -c, 4)            # per contract, for the rule's ROI
    df.to_csv(LOG, index=False)
    done = df[df.result.isin(["won", "lost"])]
    if len(done):
        for th, g in done.groupby("threshold"):
            unit = pd.to_numeric(g.get("unit", pd.Series(dtype=float)), errors="coerce").fillna(0)
            print(f"\nLIVE FIRST-TOUCH RULE at {th:.2f} (paper): {len(g)} tickers, {int((g.result == 'won').sum())} won, "
                  f"avg price {g.price.mean():.3f}, ROI {unit.sum() / g.price.sum():+.2%} (a game appears as two tickers)")
        money = done[pd.to_numeric(done.stake, errors="coerce").fillna(0) > 0]
        if len(money):
            from . import fund
            b = fund.load_cfg()["bankroll"]
            pnl = pd.to_numeric(money.pnl).sum()
            print(f"PAPER FUND, core sleeve: {len(money)} bets, {int((money.result == 'won').sum())} won, "
                  f"P&L ${pnl:+.2f} -> paper bankroll ${b + pnl:,.2f} (started ${b:,.2f})")
