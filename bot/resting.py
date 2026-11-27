"""Resting orders: instead of paying the ask, post a bid a few cents under it and wait. Would it have been filled
before the game, and what did the filled ones return?

Why it matters: the watch list is full of bets that are positive at our estimate but not after the taker fee and the
spread. A resting order on a player prop pays no fee (Kalshi fee type `quadratic`: takers only) and buys below the ask.
The catch is adverse selection: a resting bid is filled when somebody wants out at that price, which is more often
when the bet is about to lose. Only the fills can say which effect wins.

Method, from every real pre-game fill on file (both seasons), per prop market and side:
  1. the first time a taker paid the ask in the window from 3 hours to 30 minutes before the start is "the ask we saw" (a)
  2. we post a bid at a - d cents and leave it until the game starts
  3. it counts as FILLED only if a later pre-game trade printed BELOW our bid on our side (anyone selling lower would have
     hit our better bid first). Trades exactly at our price are reported separately as "touched": queue position unknown.
  4. filled bets are graded at our bid with no fee; the comparison is paying the ask at step 1 with the taker fee.

`python -m bot resting` prints the tables and writes data/resting_orders.json, which the QuantSport chain quotes.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd

from . import config
from .props_study import STAT

FILLS = config.DATA_DIR / "prop_pre_fills.parquet"       # compact pre-game prop fills, built from the raw files; not in git
TABLE = config.DATA_DIR / "resting_orders.json"
FEE = 0.07
STEPS = (1, 2, 3, 5)                                     # cents under the ask
LO, HI = 0.20, 0.85                                      # the same price range the leg categories are measured on
WINDOW, LATEST = -180.0, -30.0                           # minutes before the start: the ask must be seen by 30 minutes out


def build(rebuild: bool = False) -> pd.DataFrame:
    """Every pre-game fill (last 3 hours) on a settled prop market, both seasons. Needs the raw fill files."""
    import functools

    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    from .trades import PRIOR_DIR, PRIOR_RESULTS, RESULTS, TRADES_DIR
    if FILLS.exists() and not rebuild:
        return pd.read_parquet(FILLS)
    series = list(STAT)
    flt = functools.reduce(lambda a, b: a | b, [pc.starts_with(pc.field("ticker"), s + "-") for s in series])
    parts = []
    for season, res_f, tdir in (("last season", PRIOR_RESULTS, PRIOR_DIR), ("this season", RESULTS, TRADES_DIR)):
        if not res_f.exists() or not tdir.exists():
            continue
        r = pd.read_parquet(res_f)
        r = r[r.series.isin(series) & r.result.isin(["yes", "no"])]
        tip = r.set_index("ticker").tipoff
        for f in sorted(tdir.glob("2*.parquet")):
            try:
                t = ds.dataset(f).to_table(columns=["ticker", "yes_price", "taker_side", "created_time", "count"], filter=flt).to_pandas()
            except Exception as e:
                print(f"  ! {f.name}: {str(e)[:70]}")
                continue
            t = t[t.ticker.isin(tip.index)]
            if t.empty:
                continue
            t["mins"] = ((t.created_time - t.ticker.map(tip)).dt.total_seconds() / 60).astype("float32")
            t = t[(t.mins >= WINDOW) & (t.mins < 0)]
            if len(t):
                parts.append(t[["ticker", "mins", "yes_price", "taker_side", "count"]].assign(season=season))
            del t
            gc.collect()
    d = pd.concat(parts, ignore_index=True)
    d["yes_price"] = d.yes_price.astype("float32")
    d.to_parquet(FILLS, index=False)
    return d


def orders(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per market and side: the ask first seen, and the lowest price our side traded at afterwards."""
    from .trades import PRIOR_RESULTS, RESULTS
    d = d if d is not None else build()
    res = pd.concat([pd.read_parquet(f)[["ticker", "series", "league", "game", "result"]] for f in (PRIOR_RESULTS, RESULTS) if f.exists()]).drop_duplicates("ticker")
    out = []
    for side in ("no", "yes"):
        # price of OUR side in every trade; a taker on our side paid the ask, a taker on the other side hit a bid on our side
        yp = (d.yes_price.astype(float) * 1000).round()                          # tenths of a cent as whole numbers: no float noise in comparisons
        x = d.assign(px=((1000 - yp) if side == "no" else yp) / 1000)
        ref = x[(x.taker_side == side) & (x.mins <= LATEST)].sort_values("mins").drop_duplicates("ticker")[["ticker", "mins", "px", "season"]]
        ref = ref.rename(columns={"mins": "t0", "px": "ask"})
        later = x[x.taker_side != side].merge(ref[["ticker", "t0"]], on="ticker")
        later = later[later.mins > later.t0]
        low = later.groupby("ticker").px.min().rename("low")                     # lowest bid on our side that got hit after we looked
        o = ref.merge(low, on="ticker", how="left").merge(res, on="ticker")
        o["side"] = side
        o["win"] = (o.result == side).astype(float)
        out.append(o)
    o = pd.concat(out, ignore_index=True)
    return o[(o.ask >= LO) & (o.ask <= HI)].reset_index(drop=True)


def _ratio(net: pd.Series, cost: pd.Series, game: pd.Series) -> tuple[float, float]:
    """Return per $1 staked and its standard error counted by game."""
    if len(net) < 2 or cost.sum() <= 0:
        return float("nan"), float("nan")
    r = float(net.sum() / cost.sum())
    resid = (net - r * cost).groupby(game).sum()
    G = len(resid)
    return r, (float(np.sqrt(G / (G - 1) * (resid ** 2).sum()) / cost.sum()) if G > 1 else float("nan"))


def stat(o: pd.DataFrame, step: int) -> dict:
    """One group of orders at one distance under the ask."""
    bid = (o.ask * 1000 - step * 10).round() / 1000
    ok = bid >= 0.05
    o, bid = o[ok], bid[ok]
    filled = (o.low * 1000).round() < (bid * 1000).round()
    touched = (o.low * 1000).round() <= (bid * 1000).round()
    f = o[filled]
    ret, se = _ratio(f.win - bid[filled], bid[filled], f.game)
    ret_t, se_t = _ratio(o[touched].win - bid[touched], bid[touched], o[touched].game)
    taker_cost = o.ask + FEE * o.ask * (1 - o.ask)
    tk, tk_se = _ratio(o.win - taker_cost, taker_cost, o.game)
    # adverse selection: did the filled ones win less often, relative to the ask, than the ones nobody sold us?
    gap_f = float((f.win - f.ask).mean()) if len(f) else float("nan")
    gap_u = float((o[~filled].win - o[~filled].ask).mean()) if (~filled).any() else float("nan")
    return dict(step=step, orders=int(len(o)), games=int(o.game.nunique()), fill=float(filled.mean()) if len(o) else float("nan"),
                touch=float(touched.mean()) if len(o) else float("nan"), filled=int(filled.sum()), ret=ret, se=se, ret_touch=ret_t, se_touch=se_t,
                taker=tk, taker_se=tk_se, gap_filled=gap_f, gap_unfilled=gap_u,
                per_order=float(((o.win - bid) * filled).sum() / len(o)) if len(o) else float("nan"))


def categories(o: pd.DataFrame) -> dict[str, set]:
    """Tickers in each graded leg category (unders), from the same history the categories are graded on."""
    from . import legs
    out = {}
    for sport in ("NBA", "NFL"):
        try:
            h = legs.history(sport)
        except Exception as e:
            print(f"  ! {sport} categories unavailable: {str(e)[:80]}")
            continue
        lab = legs.labels(sport)
        for name, f in legs.CATS[sport].items():
            if lab.get(name, {}).get("label") in ("PREFER", "lean prefer", "AVOID", "lean avoid"):
                try:
                    out[f"{sport}: {name} ({lab[name]['label']})"] = set(h[f(h).fillna(False)].ticker)
                except Exception:
                    pass
    return out


def study(o: pd.DataFrame | None = None) -> dict:
    o = o if o is not None else orders()
    groups = {}
    for (lg, side), g in o.groupby(["league", "side"]):
        groups[f"{lg} all prop {'unders' if side == 'no' else 'overs'}"] = g
    no = o[o.side == "no"]
    for name, ticks in categories(o).items():
        groups[name] = no[no.ticker.isin(ticks)]
    rows = []
    for name, g in groups.items():
        for season, s in [("both", g)] + [(k, v) for k, v in g.groupby("season")]:
            for step in STEPS:
                if len(s) >= 50:
                    rows.append(dict(group=name, season=season) | stat(s, step))
    t = pd.DataFrame(rows)
    out = dict(made=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), method="first ask seen 180-30 minutes before the start; bid = ask - step; "
               "filled = a later pre-game trade printed below the bid; graded at the bid, no fee (prop fee type: takers only)",
               rows=json.loads(t.replace({np.nan: None}).to_json(orient="records")))
    TABLE.write_text(json.dumps(out, indent=1))
    return out | dict(table=t)


def lookup(group: str, step: int = 2) -> dict | None:
    """The pooled row for one group at one step, for the chain's WAIT text."""
    if not TABLE.exists():
        return None
    for r in json.loads(TABLE.read_text()).get("rows", []):
        if r["group"] == group and r["season"] == "both" and r["step"] == step:
            return r
    return None


def run(args=None) -> None:
    rebuild = bool(args and "--rebuild" in args)
    d = build(rebuild)
    o = orders(d)
    print(f"\nRESTING ORDERS on player props: {len(d):,} pre-game fills, {len(o):,} market-sides where a taker paid an ask of {LO * 100:.0f}-{HI * 100:.0f}c "
          f"between 3 hours and 30 minutes before the start.")
    print("We bid `step` cents under that ask and wait. fill = a later pre-game trade printed BELOW our bid (certain fill); touch = at or below (queue unknown).")
    print("ret = return per $1 on the filled bids, no fee; taker = paying the ask at once with the fee, same markets. +/- is one standard error by game.")
    print("gap_filled / gap_unfilled = how often it won minus the ask, for bids that filled / did not: the difference is adverse selection.\n")
    t = study(o)["table"]
    pd.set_option("display.width", 250)
    show = t[t.season == "both"].copy()
    for c in ("fill", "touch", "ret", "se", "ret_touch", "taker", "taker_se", "gap_filled", "gap_unfilled"):
        show[c] = show[c].map(lambda v: "" if v != v else f"{v:+.1%}" if c not in ("fill", "touch", "se", "taker_se") else f"{v:.1%}")
    print(show[["group", "step", "orders", "games", "fill", "touch", "filled", "ret", "se", "ret_touch", "taker", "taker_se", "gap_filled", "gap_unfilled"]].to_string(index=False))
    print("\nBy season, 2 cents under the ask (a result only counts if it holds in both):")
    s2 = t[(t.step == 2) & (t.season != "both")].copy()
    s2["z"] = s2.ret / s2.se
    for c in ("fill", "ret", "se", "taker"):
        s2[c] = s2[c].map(lambda v: "" if v != v else f"{v:+.1%}" if c in ("ret", "taker") else f"{v:.1%}")
    print(s2[["group", "season", "orders", "games", "fill", "filled", "ret", "se", "z", "taker"]].round(1).to_string(index=False))
