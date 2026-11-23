"""Download Kalshi's own settled markets + pre-game prices -> data/history/<SERIES>.parquet

For every settled market we keep the result and the last hourly bid/ask candle that closed at or
before tip-off. That gives a real, no-lookahead answer to "what did Kalshi charge before the game,
and what happened?" for every leg type — the ground truth every strategy here is tested against.
Resumable: series already on disk are only topped up with newer events.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from .kalshi import Kalshi, data_client

import os
HIST_DIR = config.DATA_DIR / "history"
HIST_DIR.mkdir(parents=True, exist_ok=True)
# v2 files add an "early" snapshot (>= 3h before tip-off); they replace v1 files series by series
HIST_V2 = config.DATA_DIR / "history_v2"
HIST_V2.mkdir(parents=True, exist_ok=True)

NBA_SERIES = ["KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBASPREAD", "KXNBATOTAL", "KXNBAPTS", "KXNBAGAME",
              "KXNBAPRA", "KXNBASTL", "KXNBABLK", "KXNBASTOCK", "KXNBA2D", "KXNBA3D", "KXNBATEAMTOTAL",
              "KXNBA1H", "KXNBA1HSPREAD", "KXNBA1HTOTAL", "KXNBA1Q", "KXNBA1QSPREAD", "KXNBA1QTOTAL"]
NFL_SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL", "KXNFLTD", "KXNFLRECYDS",
              "KXNFLRSHYDS", "KXNFLPASSYDS", "KXNFLPASSTDS", "KXNFLREC", "KXNFL1H", "KXNFL1HSPREAD",
              "KXNFL1HTOTAL", "KXNFLRSHATT", "KXNFLPASSATT", "KXNFLPASSCOMP"]
# occurrence_datetime is the expected END of the game; subtract this to get the scheduled start
START_OFFSET = {"NBA": timedelta(hours=3), "NFL": timedelta(hours=3)}  # both verified against live trading


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def list_settled(k: Kalshi, series: str, min_close: datetime) -> list[dict]:
    return list(k.paginate("/markets", "markets", {
        "series_ticker": series, "status": "settled", "limit": 1000, "mve_filter": "exclude",
        "min_close_ts": int(min_close.timestamp())}, max_pages=400))


def _candles(k: Kalshi, tickers: list[str], start: datetime, end: datetime) -> dict:
    d = k.request("GET", "/markets/candlesticks", params={
        "market_tickers": ",".join(tickers), "start_ts": int(start.timestamp()),
        "end_ts": int(end.timestamp()), "period_interval": 60})
    return {m["market_ticker"]: m.get("candlesticks") or [] for m in d.get("markets") or []}


def download_series(k: Kalshi, series: str, since: datetime, league: str = "NBA", workers: int = 1) -> pd.DataFrame:
    path = HIST_V2 / f"{series}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and len(old):
        since = max(since, pd.Timestamp(old.close_time.max()).to_pydatetime() - timedelta(days=1))
    ms = list_settled(k, series, since)
    rows = {}
    for m in ms:
        occ = _ts(m.get("occurrence_datetime") or m.get("expected_expiration_time"))
        if not occ:
            continue
        tip = occ - START_OFFSET[league]
        rows[m["ticker"]] = dict(
            ticker=m["ticker"], event_ticker=m["event_ticker"], series=series, league=league,
            game=m["event_ticker"].split("-")[1] if "-" in m["event_ticker"] else "", title=m.get("title", ""),
            result=m.get("result", ""), settle_value=_f(m.get("settlement_value_dollars")),
            floor_strike=_f(m.get("floor_strike")), strike_type=m.get("strike_type", ""),
            open_time=_ts(m.get("open_time")), close_time=_ts(m.get("close_time")), tipoff=tip,
            volume=_f(m.get("volume_fp")), pre_bid=None, pre_ask=None, pre_last=None, pre_ts=None,
            live1_bid=None, live1_ask=None, live2_bid=None, live2_ask=None, early_bid=None, early_ask=None)
    # Candle calls cost 50 rate-limit tokens each regardless of size, so pack 100 markets per call:
    # sort by tip-off and let one call span every tip-off in its chunk (same evening).
    # only markets listed before tip-off (in-game re-strikes are skipped: thousands per night)
    todo = sorted((r for r in rows.values() if r["open_time"] and r["open_time"] < r["tipoff"]),
                  key=lambda r: r["tipoff"])
    jobs, cur = [], []
    for r in todo:
        if cur and (len(cur) >= 100 or r["tipoff"] - cur[0]["tipoff"] > timedelta(hours=12)):
            jobs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        jobs.append(cur)

    def work(job):
        lo, hi = job[0]["tipoff"], job[-1]["tipoff"]
        for attempt in range(3):
            try:
                return _candles(k, [r["ticker"] for r in job], lo - timedelta(hours=6),
                                hi + timedelta(hours=2, minutes=5))
            except Exception:
                time.sleep(2 * (attempt + 1))
        return {}

    def save(upto=None):
        done = [r for r in rows.values() if upto is None or r["tipoff"] <= upto]
        df = pd.DataFrame(done)
        if old is not None and len(old):
            df = pd.concat([old[~old.ticker.isin(df.ticker)], df], ignore_index=True) if len(df) else old
        if len(df):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
        return df

    for ji, job in enumerate(jobs):
        res = work(job)
        for t, allc in res.items():
            r = rows[t]
            cutoff = r["tipoff"].timestamp()
            for name, hrs in (("live1", 1), ("live2", 2)):  # in-game snapshots ~1h and ~2h after tip
                lc = [c for c in allc if cutoff + (hrs - 1) * 3600 + 60 < c["end_period_ts"] <= cutoff + hrs * 3600 + 60]
                if lc:
                    r[name + "_bid"] = _f(lc[-1].get("yes_bid", {}).get("close_dollars"))
                    r[name + "_ask"] = _f(lc[-1].get("yes_ask", {}).get("close_dollars"))
            ec = [c for c in allc if c["end_period_ts"] <= cutoff - 3 * 3600 + 60]   # >= 3h before tip-off
            if ec:
                r["early_bid"] = _f(ec[-1].get("yes_bid", {}).get("close_dollars"))
                r["early_ask"] = _f(ec[-1].get("yes_ask", {}).get("close_dollars"))
            cs = [c for c in allc if c["end_period_ts"] <= cutoff + 60]
            if not cs:
                continue
            c = cs[-1]
            r["pre_bid"] = _f(c.get("yes_bid", {}).get("close_dollars"))
            r["pre_ask"] = _f(c.get("yes_ask", {}).get("close_dollars"))
            p = c.get("price", {})
            r["pre_last"] = _f(p.get("close_dollars") or p.get("previous_dollars"))
            r["pre_ts"] = datetime.fromtimestamp(c["end_period_ts"], timezone.utc)
        if ji % 15 == 14 and ji + 1 < len(jobs):   # checkpoint: a restart resumes from here
            save(upto=jobs[ji][-1]["tipoff"] - timedelta(hours=13))
    return save()

def download(days: int = 45, leagues=("NBA", "NFL")):
    k = data_client()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    for league in leagues:
        for s in (NBA_SERIES if league == "NBA" else NFL_SERIES):
            t0 = time.time()
            try:
                df = download_series(k, s, since, league)
                priced = int(df.pre_ask.notna().sum()) if len(df) else 0
                print(f"{time.strftime('%H:%M')} {s:16s} {len(df):7d} settled markets, {priced:7d} with pre-game price  ({time.time() - t0:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"{s:16s} FAILED: {e}", flush=True)


def load(league: str | None = None) -> pd.DataFrame:
    v2 = {f.name: f for f in HIST_V2.glob("*.parquet")}
    fs = sorted(list(v2.values()) + [f for f in HIST_DIR.glob("*.parquet") if f.name not in v2])
    if not fs:
        raise SystemExit("no history yet: run `python -m bot history`")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return df[df.league == league] if league else df
