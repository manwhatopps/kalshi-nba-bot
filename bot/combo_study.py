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


# --- what buyers actually paid: every fill of a sampled combo (public, no key needed) -----------------
PRICES = config.DATA_DIR / "combo_prices.parquet"


def pick_for_pricing(n_per_bucket: int = 220, seed: int = 11) -> list[str]:
    """Settled sampled combos, the same number from every leg-count bucket, in random order."""
    d = pd.read_parquet(SAMPLE)
    d = d[d.result.isin(["yes", "no"])]
    b = pd.cut(d.n_legs, [1, 2, 3, 4, 5, 6, 8, 10, 15, 25, 100])
    parts = [g.sample(min(len(g), n_per_bucket), random_state=seed) for _, g in d.groupby(b, observed=True)]
    return list(pd.concat(parts).sample(frac=1, random_state=seed).ticker)


def fetch_prices(k: Kalshi, tickers: list[str], out=PRICES, pause: float = 3.2, save_every: int = 25) -> pd.DataFrame:
    """One call per combo (each costs 50 of the 600 rate-limit tokens). Resumable: combos already in `out` are skipped."""
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["ticker", "created_time", "yes_price", "count", "taker_side"])
    done, rows = set(have.ticker), []
    todo = [t for t in tickers if t not in done]
    print(f"combo prices: {len(done)} on file, {len(todo)} to fetch", flush=True)
    for i, t in enumerate(todo, 1):
        try:
            tr = k.request("GET", "/markets/trades", params={"ticker": t, "limit": 1000}).get("trades") or []
        except Exception as e:
            print(f"  ! {t[:40]}: {str(e)[:80]}", flush=True)
            time.sleep(10)
            continue
        if not tr:
            rows.append(dict(ticker=t, created_time=pd.NaT, yes_price=np.nan, count=0.0, taker_side=""))
        for x in tr:
            rows.append(dict(ticker=t, created_time=_ts(x.get("created_time")), yes_price=_f(x.get("yes_price_dollars")),
                             count=_f(x.get("count_fp")), taker_side=x.get("taker_side") or ""))
        if i % save_every == 0 or i == len(todo):
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            have["created_time"] = pd.to_datetime(have.created_time, utc=True)
            have.to_parquet(out, index=False)
            rows = []
            print(f"  {i}/{len(todo)} combos priced", flush=True)
        time.sleep(pause)
    return have


def leg_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per leg of every sampled combo, with the leg's last pre-game price where a prop print exists."""
    L = d.legs.str.split("|")
    legs = pd.DataFrame({"i": np.repeat(d.index.values, L.str.len()), "leg": np.concatenate(L.values)})
    legs[["ticker", "side"]] = legs.leg.str.rsplit(":", n=1, expand=True)
    legs["series"] = legs.ticker.str.split("-").str[0]
    legs["event"] = legs.ticker.str.split("-").str[:2].str.join("-")
    legs["game"] = legs.ticker.str.split("-").str[1]
    fp = config.DATA_DIR / "prop_pre.parquet"
    if fp.exists():
        pv = pd.read_parquet(fp).pivot_table(index="ticker", columns="taker_side", values="cost", aggfunc="last")
        legs = legs.join(pv, on="ticker")
        own = np.where(legs.side == "yes", legs["yes"], legs["no"])
        other = np.where(legs.side == "yes", 1 - legs["no"], 1 - legs["yes"]) + 0.01
        legs["price"] = pd.Series(own, index=legs.index).fillna(pd.Series(other, index=legs.index))
    else:
        legs["price"] = np.nan
    return legs


def report() -> None:
    """What real Kalshi combos look like, how often they hit against their legs' prices, and what buyers paid."""
    d = pd.read_parquet(SAMPLE)
    d = d[d.result.isin(["yes", "no"])].reset_index(drop=True)
    d["hit"] = (d.result == "yes").astype(float)
    legs = leg_table(d)
    per_ev = legs.groupby(["i", "event"]).size()
    print(f"\nREAL KALSHI COMBOS: {len(d):,} settled NBA/NFL combos created {d.open_time.min():%b %-d} to {d.open_time.max():%b %-d}, {len(legs):,} legs")
    print(f"  legs bought as NO: {(legs.side == 'no').mean():.1%}   combos that are all NO: {(legs.assign(no=legs.side == 'no').groupby('i').no.mean() == 1).mean():.1%}   "
          f"combos with 2+ legs from one market group: {(per_ev.groupby('i').max() > 1).mean():.1%} (most seen: {per_ev.max()})")
    b = pd.cut(d.n_legs, [1, 2, 3, 4, 5, 6, 8, 10, 15, 25, 100])
    print("\n  How often they hit, by number of legs:")
    print(d.groupby(b, observed=True).agg(combos=("hit", "size"), hit_rate=("hit", "mean")).round(4).to_string())

    g = legs.groupby("i").agg(fair=("price", lambda s: s.prod(skipna=False)), n=("price", "size"), games=("game", "nunique"))
    g = g.dropna(subset=["fair"])
    g["hit"], g["per_game"] = d.loc[g.index, "hit"].values, g.n / g.games
    print(f"\n  Hit rate against the product of the legs' own pre-game prices ({len(g):,} combos whose legs are all priced props).")
    print("  ratio 1.00 = the legs behave as independent; above 1 = they move together, so the product understates the chance.")
    t = g.groupby(pd.cut(g.per_game, [0, 1, 1.5, 2, 3, 5, 30]), observed=True).agg(combos=("hit", "size"), hit_rate=("hit", "mean"), product=("fair", "mean"))
    t["ratio"] = t.hit_rate / t["product"]
    print(t.rename_axis("legs per game").round(4).to_string())

    if not PRICES.exists():
        print("\n  Prices paid: not fetched yet (python -m bot combos --prices).")
        return
    p = pd.read_parquet(PRICES)
    p = p[(p.taker_side == "yes") & p.yes_price.gt(0)].sort_values("created_time")
    first = p.groupby("ticker").agg(paid=("yes_price", "first"), contracts=("count", "first"))
    m = d.join(first, on="ticker").dropna(subset=["paid"])
    m["cost"] = m.paid + 0.07 * m.paid * (1 - m.paid)
    print(f"\n  WHAT BUYERS PAID ({len(m):,} combos priced so far; the same number sampled from each leg-count bucket).")
    print("  keep = hit rate / average cost with fee: 1.00 is fair, 0.80 means buyers got back 80c per $1.")
    bb = pd.cut(m.n_legs, [1, 3, 5, 8, 15, 100])
    t = m.groupby(bb, observed=True).agg(combos=("hit", "size"), hit_rate=("hit", "mean"), avg_cost=("cost", "mean"), wins=("hit", "sum"))
    t["keep"] = t.hit_rate / t.avg_cost
    t["keep_se"] = np.sqrt(t.hit_rate * (1 - t.hit_rate) / t.combos) / t.avg_cost
    print(t.round(4).to_string())
    tot = m.hit.mean() / m.cost.mean()
    print(f"  all priced combos: hit {m.hit.mean():.2%}, cost {m.cost.mean():.2%}, keep {tot:.2f} "
          f"(+/- {np.sqrt(m.hit.mean() * (1 - m.hit.mean()) / len(m)) / m.cost.mean():.2f})")
    gp = g.join(m[["paid", "cost"]], how="inner")
    if len(gp) >= 30:
        r = gp.cost / gp.fair
        print(f"  Against the legs' own prices ({len(gp)} combos): the median buyer paid {r.median():.2f}x the product of the leg prices; "
              f"those combos hit {gp.hit.mean():.2%} vs product {gp.fair.mean():.2%} and cost {gp.cost.mean():.2%}.")
