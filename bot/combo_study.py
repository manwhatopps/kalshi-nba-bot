"""How have REAL Kalshi combos done, by number of legs and by the price buyers paid?

Sample: Kalshi lists multivariate (combo) events ordered by an internal hash, which is unrelated to
when a combo was made or whether it won - so paging through that list is a fair random sample.
For every sampled combo whose legs are all NBA/NFL markets we take the result and the price of
its first traded hour (what the first buyers paid), plus each leg's own result and start time.

  keep = hit rate / average price paid.  1.00 = fair.  Below 1 = buyers overpaid by that much.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .history import _candles, _f, _ts
from .kalshi import Kalshi, data_client

SAMPLE = config.DATA_DIR / "combo_sample.parquet"
SPORT = ("KXNBA", "KXNFL")


def sample_created(k: Kalshi, start: datetime, end: datetime, per_day: int = 10, seed: int = 5) -> pd.DataFrame:
    """Random moments between 15:00 and 04:00 UTC each day; at each, the 1,000 combos created just
    before it (about 3 seconds' worth). Creation time is unrelated to outcome, so this is unbiased."""
    rng = np.random.default_rng(seed)
    rows, day = [], start
    while day <= end:
        for off in rng.uniform(15 * 3600, 28 * 3600, per_day):
            t = int(day.timestamp() + off)
            try:
                d = k.request("GET", "/markets", params={"mve_filter": "only", "limit": 1000,
                                                         "min_created_ts": t - 600, "max_created_ts": t})
            except Exception as e:
                print(f"  ! {str(e)[:100]}", flush=True)
                continue
            for m in d.get("markets") or []:
                legs = m.get("mve_selected_legs") or []
                if m.get("result") not in ("yes", "no") or not legs or float(m.get("volume_fp") or 0) <= 0:
                    continue
                if not all(l["market_ticker"].startswith(SPORT) for l in legs):
                    continue
                rows.append(dict(ticker=m["ticker"], result=m["result"], volume=_f(m.get("volume_fp")),
                                 open_time=_ts(m.get("open_time")), close_time=_ts(m.get("close_time")),
                                 n_legs=len(legs), legs="|".join(f"{l['market_ticker']}:{l['side']}" for l in legs)))
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows).drop_duplicates("ticker")


def add_prices(k: Kalshi, df: pd.DataFrame) -> pd.DataFrame:
    """First traded hour of each combo: mean trade price and volume."""
    df = df.sort_values("open_time").reset_index(drop=True)
    price, vol, when = {}, {}, {}
    for i in range(0, len(df), 100):
        chunk = df.iloc[i:i + 100]
        lo, hi = chunk.open_time.min(), chunk.close_time.max()
        if (hi - lo).total_seconds() > 14 * 86400:     # keep each request's window sane
            hi = lo + pd.Timedelta(days=14)
        try:
            cs = _candles(k, list(chunk.ticker), lo.to_pydatetime() - pd.Timedelta(hours=1), hi.to_pydatetime())
        except Exception as e:
            print(f"  ! candles failed: {str(e)[:120]}", flush=True)
            continue
        for t, cl in cs.items():
            for c in cl:
                if float(c.get("volume_fp") or 0) > 0:
                    p = c.get("price", {})
                    price[t] = _f(p.get("mean_dollars") or p.get("open_dollars") or p.get("close_dollars"))
                    vol[t] = float(c["volume_fp"])
                    when[t] = datetime.fromtimestamp(c["end_period_ts"] - 3600, timezone.utc)
                    break
    df["price"] = df.ticker.map(price)
    df["first_volume"] = df.ticker.map(vol)
    df["first_trade_hour"] = df.ticker.map(when)
    return df


def add_legs(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    r = results.set_index("ticker")
    res, tip, grp = r.result.to_dict(), r.tipoff.to_dict(), None
    from .report_trades import group
    first_tip, hits, known, games, kinds = [], [], [], [], []
    for legs in df.legs:
        pairs = [x.rsplit(":", 1) for x in legs.split("|")]
        tips = [tip[t] for t, _ in pairs if t in tip]
        first_tip.append(min(tips) if tips else pd.NaT)
        hits.append(sum(res.get(t) == s for t, s in pairs))
        known.append(sum(res.get(t) in ("yes", "no") for t, _ in pairs))
        games.append(len({t.split("-")[1] for t, _ in pairs if "-" in t}))
        kinds.append(",".join(sorted({group(t.split("-")[0])[2:] for t, _ in pairs})))
    return df.assign(first_tip=first_tip, legs_hit=hits, legs_known=known, n_games=games, kinds=kinds)


def table(df: pd.DataFrame, by: str, bins=None, labels=None) -> pd.DataFrame:
    d = df.dropna(subset=["price"]).copy()
    d["win"] = (d.result == "yes").astype(float)
    key = pd.cut(d[by], bins, labels=labels, right=False) if bins is not None else d[by]
    rows = []
    for kx, g in d.groupby(key, observed=True):
        if len(g) < 30:
            continue
        keep = g.win.mean() / g.price.mean()
        # bootstrap-free SE of the ratio (delta method on win; price treated as known)
        se = np.sqrt(g.win.mean() * (1 - g.win.mean()) / len(g)) / g.price.mean()
        rows.append({by: kx, "combos": len(g), "avg_legs": g.n_legs.mean(), "avg_price": g.price.mean(),
                     "avg_payout_x": (1 / g.price).mean(), "hit_rate": g.win.mean(), "keep": keep, "keep_se": se,
                     "roi_per_$": (g.win / g.price).mean() - 1})
    return pd.DataFrame(rows)
