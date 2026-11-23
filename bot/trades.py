"""Every NBA/NFL trade on Kalshi, with who was the taker.

Source: the Hugging Face dataset TimeSeventeen/Kalshi (one parquet of ALL Kalshi fills per UTC day,
~550 MB, no licence stated - used here for private analysis only, never redistributed). Each day is
downloaded, filtered to KXNBA*/KXNFL* tickers, saved compactly under data/trades/, and the raw file
deleted. Results and tip-off times come from Kalshi's own settled-market listing (data/results.parquet).

Every fill has two sides, so each one is two real bets with real prices:
  taker  paid the quoted price on `taker_side`
  maker  had a resting order filled on the other side at (1 - that price)
That lets us measure, on what actually traded, how each side did - including the adverse selection
a resting order suffers, which quote snapshots cannot show.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import config
from .history import START_OFFSET, _f, _ts
from .kalshi import Kalshi, data_client

HF = "https://huggingface.co/datasets/TimeSeventeen/Kalshi/resolve/main/daily/{d}.parquet"
TRADES_DIR = config.DATA_DIR / "trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = config.DATA_DIR / "results.parquet"
TAKER_FEE, MAKER_FEE = 0.07, 0.0175


def fetch_day(d: date) -> int | None:
    """Download one UTC day, keep NBA/NFL fills. Returns rows kept (None if the day isn't published)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    out = TRADES_DIR / f"{d:%Y_%m_%d}.parquet"
    if out.exists():
        return -1
    raw = TRADES_DIR / f"_raw_{d:%Y_%m_%d}.parquet"
    r = requests.get(HF.format(d=f"{d:%Y_%m_%d}"), stream=True, timeout=600)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with open(raw, "wb") as f:
        for chunk in r.iter_content(1 << 22):
            f.write(chunk)
    t = ds.dataset(raw).to_table(
        columns=["ticker", "count", "yes_price", "taker_side", "created_time"],
        filter=pc.starts_with(pc.field("ticker"), "KXNBA") | pc.starts_with(pc.field("ticker"), "KXNFL"))
    df = t.to_pandas()
    df["yes_price"] = (df.yes_price / 100).astype("float32")
    df["count"] = df["count"].astype("float32")
    df.to_parquet(out, index=False)
    raw.unlink()
    return len(df)


def download(start: date, end: date | None = None):
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    d = start
    while d <= end:
        t0 = time.time()
        try:
            n = fetch_day(d)
            msg = "already have" if n == -1 else ("not published yet" if n is None else f"{n:,} NBA/NFL fills")
        except Exception as e:
            msg = f"FAILED {e}"
        print(f"{d}  {msg}  ({time.time() - t0:.0f}s)", flush=True)
        d += timedelta(days=1)


def load_trades() -> pd.DataFrame:
    fs = sorted(f for f in TRADES_DIR.glob("2*.parquet"))
    if not fs:
        raise SystemExit("no trades yet: run `python -m bot trades --download`")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def fetch_results(k: Kalshi, series_list: list[str], since: datetime) -> pd.DataFrame:
    """Result + scheduled start for every settled market of the given series (cheap listing calls)."""
    old = pd.read_parquet(RESULTS) if RESULTS.exists() else pd.DataFrame()
    rows = []
    for s in series_list:
        s_since = since
        if len(old) and (old.series == s).any():
            s_since = max(since, pd.Timestamp(old.close_time[old.series == s].max()).to_pydatetime() - timedelta(days=1))
        league = "NFL" if s.startswith("KXNFL") else "NBA"
        try:
            ms = list(k.paginate("/markets", "markets", {
                "series_ticker": s, "status": "settled", "limit": 1000, "mve_filter": "exclude",
                "min_close_ts": int(s_since.timestamp())}, max_pages=600))
        except Exception as e:
            print(f"  ! {s}: {e}", flush=True)
            continue
        for m in ms:
            occ = _ts(m.get("occurrence_datetime") or m.get("expected_expiration_time"))
            if not occ or "-" not in m["event_ticker"]:
                continue
            rows.append(dict(ticker=m["ticker"], event_ticker=m["event_ticker"], series=s, league=league,
                             game=m["event_ticker"].split("-")[1], title=m.get("title", ""),
                             result=m.get("result", ""), floor_strike=_f(m.get("floor_strike")),
                             open_time=_ts(m.get("open_time")), close_time=_ts(m.get("close_time")),
                             tipoff=occ - START_OFFSET[league]))
        print(f"  {s:18s} {len(ms):7d} settled markets", flush=True)
        df = pd.DataFrame(rows)
        if len(old):
            df = pd.concat([old[~old.ticker.isin(df.ticker)], df], ignore_index=True) if len(df) else old
        if len(df):
            df.to_parquet(RESULTS, index=False)     # checkpoint after every series
    return pd.read_parquet(RESULTS) if RESULTS.exists() else pd.DataFrame()


AGG = config.DATA_DIR / "fills_agg.parquet"
PHASES = ["early (3h+ before)", "last 3h before", "in-game", "late / after"]
KEYS = ["league", "series", "game", "phase", "cents", "taker_side", "result"]


AGG_DAYS = config.DATA_DIR / "agg_days.json"


def aggregate(results: pd.DataFrame | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Day by day (memory-safe): join fills to results and collapse to
    (market type, game, phase, price in cents, taker side, result) -> fills / contracts / capped contracts.
    Incremental: only day files not already folded in are processed, so it works on a fresh machine
    that has the committed summary but not the raw trade files."""
    import json
    results = results if results is not None else pd.read_parquet(RESULTS)
    r = results[results.result.isin(["yes", "no"])][["ticker", "series", "league", "game", "result", "tipoff"]]
    done = set() if rebuild or not (AGG.exists() and AGG_DAYS.exists()) else set(json.loads(AGG_DAYS.read_text()))
    parts = [pd.read_parquet(AGG)] if done else []
    for f in sorted(TRADES_DIR.glob("2*.parquet")):
        if f.stem in done:
            continue
        t = pd.read_parquet(f).merge(r, on="ticker", how="inner")
        done.add(f.stem)
        if t.empty:
            continue
        mins = (t.created_time - t.tipoff).dt.total_seconds() / 60
        t["phase"] = np.select([mins < -180, mins < 0, mins < 150], PHASES[:3], PHASES[3])
        t["cents"] = (t.yes_price * 100).round().astype("int16")
        t["cap100"] = t["count"].clip(upper=100)
        g = t.groupby(KEYS, observed=True).agg(n=("count", "size"), contracts=("count", "sum"), cap100=("cap100", "sum"))
        parts.append(g.reset_index())
    a = pd.concat(parts, ignore_index=True).groupby(KEYS, observed=True)[["n", "contracts", "cap100"]].sum().reset_index()
    a.to_parquet(AGG, index=False)
    AGG_DAYS.write_text(json.dumps(sorted(done)))
    return a


def load_agg() -> pd.DataFrame:
    return pd.read_parquet(AGG) if AGG.exists() else aggregate()


def with_roles(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    p = a.cents.values / 100.0
    yes_taker = (a.taker_side == "yes").values
    a["taker_cost"] = np.where(yes_taker, p, 1 - p)
    a["taker_win"] = (a.taker_side == a.result).astype(float)
    a["maker_cost"] = 1 - a.taker_cost
    a["maker_win"] = 1 - a.taker_win
    return a


def role_table(a: pd.DataFrame, role: str, weight: str = "n", by: str = "phase",
               buckets=(0.01, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99, 1.0)) -> pd.DataFrame:
    """weight: 'n' = every fill counts once (a small bettor's view), 'cap100' = contracts capped at
    100 per fill, 'contracts' = real dollars (dominated by a few whales). ROI is after Kalshi fees;
    standard errors are clustered by game."""
    cost, win, w = a[f"{role}_cost"], a[f"{role}_win"], a[weight].astype(float)
    rate = TAKER_FEE if role == "taker" else MAKER_FEE
    net = (win - cost - rate * cost * (1 - cost)) * w
    b = pd.cut(cost, list(buckets), right=False)
    rows = []
    for (key, bk), idx in a.groupby([a[by], b], observed=True).groups.items():
        n, c, ww = net.loc[idx], cost.loc[idx], w.loc[idx]
        cl = n.groupby(a.game.loc[idx]).sum()
        G, staked = len(cl), (c * ww).sum()
        se = np.sqrt(max((cl ** 2).sum() - cl.sum() ** 2 / G, 0) * G / max(G - 1, 1)) / staked if G > 1 else np.nan
        rows.append({by: key, "price": str(bk), "fills": int(a.n.loc[idx].sum()), "games": G,
                     "avg_price": staked / ww.sum(), "hit": (win.loc[idx] * ww).sum() / ww.sum(),
                     "roi": n.sum() / staked, "se": se})
    out = pd.DataFrame(rows)
    out["t"] = out.roi / out.se
    return out


def first_touch(thresholds=(0.85, 0.90, 0.95, 0.97), results: pd.DataFrame | None = None,
                phase_minutes=(0, 150), rebuild: bool = False) -> pd.DataFrame:
    """A rule you can actually follow: in each market, buy ONE contract the first time a taker
    could buy either side at >= threshold during the game (first in-game fill at that price).
    One row per (market, side, threshold) with the entry price and whether it won."""
    results = results if results is not None else pd.read_parquet(RESULTS)
    import json
    r = results[results.result.isin(["yes", "no"])][["ticker", "series", "league", "game", "result", "tipoff"]]
    FT, FT_DAYS = config.DATA_DIR / "first_touch.parquet", config.DATA_DIR / "first_touch_days.json"
    cols = ["ticker", "taker_side", "created_time", "cost", "mins", "series", "league", "game", "result", "threshold"]
    seen = set(json.loads(FT_DAYS.read_text())) if (FT.exists() and FT_DAYS.exists() and not rebuild) else set()
    best = pd.read_parquet(FT)[cols] if seen else None
    for f in sorted(TRADES_DIR.glob("2*.parquet")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        t = pd.read_parquet(f).merge(r, on="ticker", how="inner")
        mins = (t.created_time - t.tipoff).dt.total_seconds() / 60
        t = t[(mins >= phase_minutes[0]) & (mins < phase_minutes[1])]
        t["mins"] = mins
        t["cost"] = np.where(t.taker_side == "yes", t.yes_price, 1 - t.yes_price).astype(float).round(2)
        parts = []
        for th in thresholds:
            x = t[(t.cost >= th) & (t.cost <= 0.99)]
            if x.empty:
                continue
            i = x.groupby(["ticker", "taker_side"]).created_time.idxmin()
            y = x.loc[i, ["ticker", "taker_side", "created_time", "cost", "mins", "series", "league", "game", "result"]]
            y["threshold"] = th
            parts.append(y)
        if parts:
            day = pd.concat(parts)
            best = day if best is None else pd.concat([best, day])
            i = best.groupby(["ticker", "taker_side", "threshold"]).created_time.idxmin()
            best = best.loc[i].reset_index(drop=True)
    best["win"] = (best.taker_side == best.result).astype(float)
    best["net"] = best.win - best.cost - TAKER_FEE * best.cost * (1 - best.cost)
    best.to_parquet(FT, index=False)
    FT_DAYS.write_text(json.dumps(sorted(seen)))
    return best


def touch_table(ft: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in ft.groupby(by, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        cl = g.net.groupby(g.game).sum()
        G = len(cl)
        se = np.sqrt(max((cl ** 2).sum() - cl.sum() ** 2 / G, 0) * G / max(G - 1, 1)) / g.cost.sum() if G > 1 else np.nan
        rows.append(dict(zip(by, k)) | dict(bets=len(g), games=G, avg_price=g.cost.mean(), hit=g.win.mean(),
                                            losses=int((1 - g.win).sum()), roi=g.net.sum() / g.cost.sum(), se=se,
                                            avg_min_into_game=g.mins.mean()))
    out = pd.DataFrame(rows)
    out["t"] = out.roi / out.se
    return out


CALIB = config.DATA_DIR / "leg_calibration.json"
CAL_BANDS = [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.99]


def leg_calibration(a: pd.DataFrame | None = None, tau: float = 0.02) -> dict:
    """Pre-game buyers, all season: how much more (or less) often each kind of leg hit than it cost,
    by league x market group x price band. Raw gaps are noisy, so each is shrunk toward zero in
    proportion to its uncertainty (empirical-Bayes with prior sd `tau`): thin evidence -> ~0."""
    import json
    from .report_trades import group
    a = with_roles(a if a is not None else load_agg())
    a = a[a.phase.isin(PHASES[:2])].assign(group=lambda d: d.series.map(group))
    out = {}
    b = pd.cut(a.taker_cost, CAL_BANDS, right=False)
    w = a.n.astype(float)
    raw = (a.taker_win - a.taker_cost) * w
    for (lg, g, bk), idx in a.groupby([a.league, a.group, b], observed=True).groups.items():
        ww = w.loc[idx]
        if ww.sum() < 2000:
            continue
        cl = raw.loc[idx].groupby(a.game.loc[idx]).sum()
        G = len(cl)
        se = np.sqrt(max((cl ** 2).sum() - cl.sum() ** 2 / G, 0) * G / max(G - 1, 1)) / ww.sum() if G > 1 else 1.0
        gap = raw.loc[idx].sum() / ww.sum()
        out[f"{lg}|{g}|{bk.left:.2f}"] = dict(gap=float(gap), se=float(se), fills=int(ww.sum()), games=int(G),
                                               shrunk=float(gap * tau ** 2 / (tau ** 2 + se ** 2)))
    CALIB.write_text(json.dumps(out, indent=1))
    return out


def expected_hit(league: str, series: str, cost: float, calib: dict) -> float:
    from .report_trades import group
    lo = max([x for x in CAL_BANDS[:-1] if x <= cost], default=None)
    c = calib.get(f"{league}|{group(series)}|{lo:.2f}") if lo is not None else None
    return float(min(max(cost + (c["shrunk"] if c else 0.0), 0.01), 0.995))
