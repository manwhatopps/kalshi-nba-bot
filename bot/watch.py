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
FIELDS = ["ts", "ticker", "side", "price", "threshold", "mins_into_game", "title", "result", "pnl"]


def _seen() -> set:
    if not LOG.exists() or LOG.stat().st_size == 0:
        return set()
    with open(LOG) as f:
        return {(r["ticker"], r["side"], r["threshold"]) for r in csv.DictReader(f)}


def run(series="KXNBAGAME", thresholds=(0.95,), hours: float = 8, every: int = 60, league="NBA"):
    k = data_client()
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
                            w.writerow(dict(ts=now.isoformat(timespec="seconds"), ticker=m["ticker"], side=side,
                                            price=round(price, 2), threshold=th, mins_into_game=round(mins),
                                            title=m.get("title", ""), result="", pnl=""))
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
            df.at[i, "pnl"] = round((1 - p if win else -p) - 0.07 * p * (1 - p), 4)
    df.to_csv(LOG, index=False)
    done = df[df.result.isin(["won", "lost"])]
    if len(done):
        pnl = pd.to_numeric(done.pnl)
        print(f"\nLIVE TOUCH RULE (paper): {len(done)} bets, {int((done.result == 'won').sum())} won, "
              f"avg price {done.price.mean():.3f}, ROI {pnl.sum() / done.price.sum():+.2%} "
              "(each game is counted twice: YES one team / NO the other)")
