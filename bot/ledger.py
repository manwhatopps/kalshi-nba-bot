"""CSV ledgers so every suggestion is a paper trade that gets graded.

data/combos.csv  - every suggested / quoted / placed combo
data/legs.csv    - every candidate leg with model, book and market probabilities
`settle` fills in results from Kalshi; `report` answers the only question that matters:
does the model beat the Kalshi price (lower Brier score) on legs that actually settled?
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from . import config

import os
_DIR = config.ROOT / os.getenv("BOT_LEDGER_DIR") if os.getenv("BOT_LEDGER_DIR") else config.DATA_DIR
_DIR.mkdir(parents=True, exist_ok=True)
COMBOS = _DIR / "combos.csv"
LEGS = _DIR / "legs.csv"


def _append(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_legs(run_id, legs):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _append(LEGS, [dict(ts=now, run_id=run_id, ticker=l.ticker, title=l.title, kind=l.kind,
                        p_model=l.p_model, p_book=l.p_book, mid=round(l.mid, 4), ask=l.yes_ask,
                        p_fair=l.p_fair, result="") for l in legs if l.p_model is not None])


def log_combo(run_id, c, status, quote_price=None, stake=None, combo_ticker="", note=""):
    _append(COMBOS, [dict(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id,
        combo_id="|".join(sorted(l.ticker for l in c.legs)), status=status, n_legs=len(c.legs),
        legs=json.dumps([l.ticker for l in c.legs]), titles=" + ".join(l.title for l in c.legs),
        p_fair=round(c.p_fair, 5), p_mid=round(c.p_mid, 5), est_price=round(c.est_price, 5),
        quote_price=quote_price, stake=stake if stake is not None else c.stake(), combo_ticker=combo_ticker,
        note=note, result="", pnl="")])


def placed_today() -> float:
    if not COMBOS.exists():
        return 0.0
    df = pd.read_csv(COMBOS)
    today = datetime.now(timezone.utc).date().isoformat()
    m = (df.status == "placed") & df.ts.str.startswith(today)
    return float(df.loc[m, "stake"].fillna(0).sum())


def settle(k):
    cache = {}
    want = set()
    if LEGS.exists():
        L0 = pd.read_csv(LEGS, dtype={"result": str}).fillna({"result": ""})
        want |= set(L0.ticker[L0.result == ""])
    if COMBOS.exists():
        C0 = pd.read_csv(COMBOS, dtype={"result": str}).fillna({"result": ""})
        for legs in C0.legs[C0.result == ""]:
            want |= {t.split(":")[0] for t in json.loads(legs)}
    want = sorted(want)
    for i in range(0, len(want), 100):  # batch lookup: 100 tickers per request
        try:
            d = k.request("GET", "/markets", params={"tickers": ",".join(want[i:i + 100]), "limit": 100})
            for m in d.get("markets") or []:
                cache[m["ticker"]] = m.get("result") or ""
        except Exception as e:
            print(f"  ! settle batch failed: {e}")
        time.sleep(0.1)

    def res(t):
        return cache.get(t, "")

    n = 0
    if LEGS.exists():
        L = pd.read_csv(LEGS, dtype={"result": str}).fillna({"result": ""})
        for i in L.index[L.result == ""]:
            r = res(L.at[i, "ticker"])
            if r in ("yes", "no"):
                L.at[i, "result"] = r
                n += 1
        L.to_csv(LEGS, index=False)
    if COMBOS.exists():
        C = pd.read_csv(COMBOS, dtype={"result": str, "pnl": str}).fillna({"result": "", "pnl": ""})
        for i in C.index[C.result == ""]:
            pairs = [(t.split(":") + ["yes"])[:2] for t in json.loads(C.at[i, "legs"])]
            rs = [(res(t), side) for t, side in pairs]   # leg wins when the market result equals its side
            if any(r in ("yes", "no") and r != side for r, side in rs):
                C.at[i, "result"] = "lost"
            elif all(r == side for r, side in rs):
                C.at[i, "result"] = "won"
            else:
                continue
            price = C.at[i, "quote_price"] if pd.notna(C.at[i, "quote_price"]) else C.at[i, "est_price"]
            stake = C.at[i, "stake"] if C.at[i, "status"] == "placed" else 1.0
            C.at[i, "pnl"] = round(stake * (1 / price - 1) if C.at[i, "result"] == "won" else -stake, 2)
            n += 1
        C.to_csv(COMBOS, index=False)
    return n


def report():
    if LEGS.exists():
        L = pd.read_csv(LEGS, dtype={"result": str}).fillna({"result": ""})
        L = L[L.result.isin(["yes", "no"])].drop_duplicates("ticker", keep="last")
        if len(L):
            y = (L.result == "yes").astype(float)
            print(f"\nLEG CALIBRATION on {len(L)} settled legs (lower Brier = better):")
            for col in ["mid", "p_model", "p_book", "p_fair"]:
                s = L[col].dropna()
                if len(s):
                    print(f"  {col:8s} Brier {((s - y[s.index]) ** 2).mean():.4f}   (n={len(s)})")
            print("  -> the model only has an edge if p_model/p_fair beat 'mid' over hundreds of legs.")
    if COMBOS.exists():
        C = pd.read_csv(COMBOS, dtype={"result": str}).fillna({"result": ""})
        C = C[C.result.isin(["won", "lost"])]
        for status, g in C.groupby("status"):
            pnl = pd.to_numeric(g.pnl, errors="coerce")
            print(f"\nCOMBOS [{status}]: {len(g)} settled, {int((g.result == 'won').sum())} won, "
                  f"expected wins {g.p_fair.sum():.1f}, P&L {pnl.sum():+.2f} "
                  f"({'per $1 paper stake' if status != 'placed' else 'dollars'})")


def log_card(run_id, card, name, p_hit, price, stake=1.0, note=""):
    """A favorites card (legs can be YES or NO) logged as one paper combo."""
    _append(COMBOS, [dict(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id,
        combo_id="|".join(sorted(f"{t}:{s}" for t, s in zip(card.ticker, card.side))), status=name,
        n_legs=len(card), legs=json.dumps([f"{t}:{s}" for t, s in zip(card.ticker, card.side)]),
        titles=" + ".join(f"{s.upper()} {t}" for t, s in zip(card.title, card.side))[:4000],
        p_fair=round(p_hit, 6), p_mid=round(price, 6), est_price=round(price, 6), quote_price=None,
        stake=stake, combo_ticker="", note=note, result="", pnl="")])
