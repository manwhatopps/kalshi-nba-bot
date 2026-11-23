"""Player props, tested across two seasons.

Every prop fill from Kalshi's prior-season archive and from this season is graded against its
result. An "angle" is market type x pick (OVER = buy YES, UNDER = buy NO) x price band x timing.
For each angle we measure, with each game counted once, how much more or less often the pick hit
than the buyer paid - separately in each season. An angle is only called real if it shows up in
BOTH seasons and survives a multiple-testing bar; one-season wonders are what the moneyline rule
turned out to be.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import trades as T

STAT = {"KXNBAPTS": "NBA points", "KXNBAREB": "NBA rebounds", "KXNBAAST": "NBA assists", "KXNBA3PT": "NBA threes",
        "KXNBAPRA": "NBA pts+reb+ast", "KXNBASTL": "NBA steals", "KXNBABLK": "NBA blocks", "KXNBA2D": "NBA double-double",
        "KXNBA3D": "NBA triple-double", "KXNBASTOCK": "NBA steals+blocks",
        "KXNFLANYTD": "NFL anytime TD", "KXNFLTD": "NFL anytime TD", "KXNFLFIRSTTD": "NFL first TD", "KXNFL2TD": "NFL 2+ TDs",
        "KXNFLRECYDS": "NFL receiving yards", "KXNFLRSHYDS": "NFL rushing yards", "KXNFLPASSYDS": "NFL passing yards",
        "KXNFLREC": "NFL receptions", "KXNFLPASSTDS": "NFL passing TDs"}
BANDS = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 0.995]
PRIOR_AGG = config.DATA_DIR / "fills_agg_prior.parquet"


def load(which: str) -> pd.DataFrame:
    a = pd.read_parquet(PRIOR_AGG if which == "archive" else T.AGG)
    a = T.with_roles(a[a.series.isin(list(STAT))])
    a["stat"] = a.series.map(STAT)
    a["pick"] = np.where(a.taker_side == "yes", "OVER", "UNDER")
    a["timing"] = np.where(a.phase.isin(T.PHASES[:2]), "pre-game", np.where(a.phase == "in-game", "in-game", "late"))
    a["band"] = pd.cut(a.taker_cost, BANDS, right=False).astype(str)
    return a[a.timing != "late"]


def game_gaps(a: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per (angle, game): that game's fill-weighted (hit - paid), plus fee-adjusted P&L sums."""
    w = a.n.astype(float)
    d = a.assign(_raw=(a.taker_win - a.taker_cost) * w,
                 _net=(a.taker_win - a.taker_cost - T.TAKER_FEE * a.taker_cost * (1 - a.taker_cost)) * w,
                 _cost=a.taker_cost * w, _hit=a.taker_win * w, _w=w)
    g = d.groupby(keys + ["game"], observed=True)[["_raw", "_net", "_cost", "_hit", "_w"]].sum().reset_index()
    g["gap"] = g._raw / g._w
    return g


def summarise(g: pd.DataFrame, keys: list[str], min_games: int = 40) -> pd.DataFrame:
    rows = []
    for k, d in g.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        if len(d) < min_games:
            continue
        se = d.gap.std() / np.sqrt(len(d))
        rows.append(dict(zip(keys, k)) | dict(fills=int(d._w.sum()), games=len(d), paid=d._cost.sum() / d._w.sum(),
                                              hit=d._hit.sum() / d._w.sum(), gap=d.gap.mean(), se=se,
                                              t=d.gap.mean() / se if se > 0 else 0.0, roi=d._net.sum() / d._cost.sum()))
    return pd.DataFrame(rows)


def two_seasons(keys: list[str], min_games: int = 40) -> pd.DataFrame:
    ga, gc = game_gaps(load("archive"), keys), game_gaps(load("current"), keys)
    A, C = summarise(ga, keys, min_games), summarise(gc, keys, min_games)
    P = summarise(pd.concat([ga, gc]), keys, min_games)
    m = A.merge(C, on=keys, suffixes=("_last", "_this")).merge(
        P[keys + ["gap", "t", "roi", "games"]].rename(columns={"gap": "gap_all", "t": "t_all", "roi": "roi_all", "games": "games_all"}), on=keys)
    k = max(len(m), 1)
    bar = float(abs(pd.Series([0]).pipe(lambda _: __import__("scipy.stats").stats.norm.ppf(0.025 / k))))   # Bonferroni
    same = np.sign(m.gap_last) == np.sign(m.gap_this)
    m["verdict"] = np.where(same & (m.t_last.abs() >= 1.64) & (m.t_this.abs() >= 1.64) & (m.t_all.abs() >= bar),
                            np.where(m.gap_all > 0, "REAL: hits MORE than priced", "REAL: hits LESS than priced"),
                            np.where(same & (m.t_all.abs() >= 2.5), "leaning, same sign both seasons", "noise"))
    m.attrs["bar"], m.attrs["angles"] = bar, k
    return m.sort_values("t_all")


def _show(m: pd.DataFrame, keys: list[str], title: str, top: int = 0):
    cols = keys + ["games_last", "paid_last", "hit_last", "gap_last", "t_last", "roi_last",
                   "games_this", "paid_this", "hit_this", "gap_this", "t_this", "roi_this", "t_all", "verdict"]
    d = m if not top else pd.concat([m.head(top), m.tail(top)])
    f = d[cols].copy()
    for c in ("gap_last", "gap_this"):
        f[c] = f[c].map(lambda v: f"{v * 100:+.1f}pt")
    for c in ("roi_last", "roi_this"):
        f[c] = f[c].map("{:+.1%}".format)
    print(f"\n{title}")
    print(f.round(3).to_string(index=False))


def run():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    a, c = load("archive"), load("current")
    print(f"\nPLAYER PROPS across two seasons. Archive: {int(a.n.sum()):,} fills, {a.game.nunique():,} games.  "
          f"This season: {int(c.n.sum()):,} fills, {c.game.nunique():,} games.")
    print("gap = hit rate minus price paid, each game counted once (pt = percentage points); roi = after Kalshi's fee.")
    m1 = two_seasons(["league", "timing", "pick", "band"])
    _show(m1, ["league", "timing", "pick", "band"], "1) OVERS vs UNDERS by price, all prop types together")
    m2 = two_seasons(["stat", "timing", "pick"])
    _show(m2, ["stat", "timing", "pick"], "2) BY PROP TYPE (all prices)")
    m3 = two_seasons(["stat", "timing", "pick", "band"], min_games=60)
    real = m3[m3.verdict.str.startswith("REAL")]
    print(f"\n3) FINE GRAIN: {m3.attrs['angles']} angles (prop type x timing x pick x price). To be called real an angle must have "
          f"the same sign in both seasons, |t| >= 1.64 in each, and pooled |t| >= {m3.attrs['bar']:.1f} (multiple-testing bar).")
    print(f"   Passed: {len(real)}.  Hit MORE than priced: {int((real.gap_all > 0).sum())}.  Hit LESS than priced: {int((real.gap_all < 0).sum())}.")
    if len(real):
        _show(real, ["stat", "timing", "pick", "band"], "   Angles that passed:")
    _show(m3, ["stat", "timing", "pick", "band"], "   The 8 most negative and 8 most positive angles (pooled):", top=8)
    buy = m3[(m3.roi_last > 0) & (m3.roi_this > 0)]
    print(f"\n   Angles where a buyer made money AFTER fees in both seasons: {len(buy)} of {len(m3)} "
          f"(by chance alone, about {len(m3) * 0.25 * 0.6:.0f} would if every angle were exactly fair before fees).")
    if len(buy):
        _show(buy.sort_values("t_all", ascending=False), ["stat", "timing", "pick", "band"], "   Those angles:")
    m3.to_parquet(config.DATA_DIR / "props_angles.parquet", index=False)
    return m1, m2, m3


# ---------------------------------------------------------------- a rule you can follow
TOUCH_BANDS = [(0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.95)]


def first_in_band(trades_dir, results: pd.DataFrame, out, sides=("no", "yes"), minutes=(0, 150)) -> pd.DataFrame:
    """For every prop market and each price band: the FIRST in-game fill where a taker bought that
    side inside the band. One contract per (market, side, band) - a rule a person could follow.
    Incremental by day file, like first_touch."""
    import json
    days_f = out.with_name(out.stem + "_days.json")
    r = results[results.result.isin(["yes", "no"]) & results.series.isin(list(STAT))][
        ["ticker", "series", "league", "game", "result", "tipoff"]]
    seen = set(json.loads(days_f.read_text())) if (out.exists() and days_f.exists()) else set()
    cols = ["ticker", "taker_side", "band", "created_time", "cost", "mins", "series", "league", "game", "result"]
    best = pd.read_parquet(out)[cols] if seen else None
    for f in sorted(trades_dir.glob("2*.parquet")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        t = pd.read_parquet(f).merge(r, on="ticker", how="inner")
        if t.empty:
            continue
        mins = (t.created_time - t.tipoff).dt.total_seconds() / 60
        t = t[(mins >= minutes[0]) & (mins < minutes[1]) & t.taker_side.isin(sides)].copy()
        t["mins"] = mins
        t["cost"] = np.where(t.taker_side == "yes", t.yes_price, 1 - t.yes_price).astype(float).round(2)
        parts = []
        for lo, hi in TOUCH_BANDS:
            x = t[(t.cost >= lo) & (t.cost < hi)]
            if x.empty:
                continue
            y = x.loc[x.groupby(["ticker", "taker_side"]).created_time.idxmin()].copy()
            y["band"] = f"{lo:.2f}-{hi:.2f}"
            parts.append(y[cols])
        if parts:
            best = pd.concat(([best] if best is not None else []) + parts)
            best = best.loc[best.groupby(["ticker", "taker_side", "band"]).created_time.idxmin()].reset_index(drop=True)
    best.to_parquet(out, index=False)
    days_f.write_text(json.dumps(sorted(seen)))
    return best


def touch_summary(ft: pd.DataFrame, keys: list[str], min_bets: int = 60) -> pd.DataFrame:
    ft = ft.assign(stat=ft.series.map(STAT), pick=np.where(ft.taker_side == "yes", "OVER", "UNDER"),
                   win=(ft.taker_side == ft.result).astype(float))
    ft["net"] = ft.win - ft.cost - T.TAKER_FEE * ft.cost * (1 - ft.cost)
    rows = []
    for k, g in ft.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        if len(g) < min_bets:
            continue
        cl = g.net.groupby(g.game).sum()
        G = len(cl)
        se = np.sqrt(max((cl ** 2).sum() - cl.sum() ** 2 / G, 0) * G / max(G - 1, 1)) / g.cost.sum()
        rows.append(dict(zip(keys, k)) | dict(bets=len(g), games=G, price=g.cost.mean(), won=g.win.mean(),
                                              roi=g.net.sum() / g.cost.sum(), se=se, t=g.net.sum() / g.cost.sum() / se,
                                              mins=g.mins.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pre-game prices per market (for card backtests)
def pregame_prices(trades_dir, results: pd.DataFrame, out, window_min: int = 180) -> pd.DataFrame:
    """Per prop market: the last price a taker paid for the OVER (yes) and for the UNDER (no) in the
    final `window_min` minutes before tip-off. Those are the leg prices a pre-game card would pay."""
    import json
    days_f = out.with_name(out.stem + "_days.json")
    r = results[results.result.isin(["yes", "no"]) & results.series.isin(list(STAT))][
        ["ticker", "series", "league", "game", "result", "tipoff"]]
    seen = set(json.loads(days_f.read_text())) if (out.exists() and days_f.exists()) else set()
    best = pd.read_parquet(out) if seen else None
    for f in sorted(trades_dir.glob("2*.parquet")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        t = pd.read_parquet(f).merge(r, on="ticker", how="inner")
        if t.empty:
            continue
        mins = (t.created_time - t.tipoff).dt.total_seconds() / 60
        t = t[(mins >= -window_min) & (mins < 0)]
        if t.empty:
            continue
        t["cost"] = np.where(t.taker_side == "yes", t.yes_price, 1 - t.yes_price).astype(float).round(2)
        y = t.loc[t.groupby(["ticker", "taker_side"]).created_time.idxmax(),
                  ["ticker", "taker_side", "created_time", "cost", "series", "league", "game", "result"]]
        best = y if best is None else pd.concat([best, y])
        best = best.loc[best.groupby(["ticker", "taker_side"]).created_time.idxmax()].reset_index(drop=True)
    best.to_parquet(out, index=False)
    days_f.write_text(json.dumps(sorted(seen)))
    return best


FAVOURED = ["NBA double-double", "NBA triple-double", "NBA blocks", "NBA steals", "NBA threes"]   # unders that leaned positive pre-game in both seasons


def load_pre() -> pd.DataFrame:
    from .favorites import slate_date
    parts = []
    for season, f in (("last season", "prop_pre_prior.parquet"), ("this season", "prop_pre.parquet")):
        fp = config.DATA_DIR / f
        if fp.exists():
            parts.append(pd.read_parquet(fp).assign(season=season))
    d = pd.concat(parts, ignore_index=True)
    d["stat"] = d.series.map(STAT)
    d["pick"] = np.where(d.taker_side == "yes", "OVER", "UNDER")
    d["win"] = (d.taker_side == d.result).astype(float)
    d["event"] = d.series + "-" + d.game                      # Kalshi allows one leg per event in a combo
    d["date"] = d.game.map(slate_date)
    return d.dropna(subset=["date"])


def card_backtest(d: pd.DataFrame, target: float, lo: float = 0.50, hi: float = 0.75, samples: int = 300,
                  max_per_game: int = 3, seed: int = 4) -> dict | None:
    """Deal random legal pre-game cards from one night's real legs until the payout target is reached;
    return how often every leg hit vs how often it needed to."""
    rng = np.random.default_rng(seed)
    pool = d[(d.cost >= lo) & (d.cost < hi)]
    hits, prices, nlegs, nights = [], [], [], 0
    for _, night in pool.groupby("date"):
        ev, gm, c, w = night.event.values, night.game.values, night.cost.values, night.win.values
        got = 0
        for _ in range(samples):
            order = rng.permutation(len(night))
            seen_e, per_g, price, ok, n = set(), {}, 1.0, True, 0
            for i in order:
                if ev[i] in seen_e or per_g.get(gm[i], 0) >= max_per_game:
                    continue
                seen_e.add(ev[i]); per_g[gm[i]] = per_g.get(gm[i], 0) + 1
                price *= c[i]; ok = ok and w[i] > 0; n += 1
                if price <= 1 / target:
                    break
            if price <= 1 / target:
                hits.append(float(ok)); prices.append(price); nlegs.append(n); got += 1
        nights += got > 0
    if not hits:
        return None
    h, p = np.array(hits), np.array(prices)
    return dict(nights=nights, cards=len(h), legs=float(np.mean(nlegs)), fair_payout=float((1 / p).mean()),
                hit_rate=float(h.mean()), needed=float(p.mean()), keep=float(h.mean() / p.mean()),
                one_in=float(1 / h.mean()) if h.mean() > 0 else float("inf"))


def card_report(samples: int = 300):
    pd.set_option("display.width", 230)
    d = load_pre()
    d = d[d.league == "NBA"]
    print(f"\nPRE-GAME LOTTERY CARDS built from real NBA prop legs priced 50-75c, both seasons "
          f"({d.date.nunique()} nights). keep = how often the card hit / how often it needed to (1.00 = fair bet).")
    rows = []
    for name, sel in (("all OVERS (what most people build)", d.pick == "OVER"),
                      ("all UNDERS", d.pick == "UNDER"),
                      ("UNDERS on double-doubles, triple-doubles, blocks, steals, threes", (d.pick == "UNDER") & d.stat.isin(FAVOURED))):
        for target in (50, 200, 1000):
            for season in ("last season", "this season"):
                r = card_backtest(d[sel & (d.season == season)], target, samples=samples)
                if r:
                    rows.append(dict(card=name, target=f"{target}x", season=season, nights=r["nights"], legs=round(r["legs"], 1),
                                     fair_payout=f"{r['fair_payout']:,.0f}x", hits_1_in=f"{r['one_in']:,.0f}",
                                     needed_1_in=f"{1 / r['needed']:,.0f}", keep=round(r["keep"], 2)))
    print(pd.DataFrame(rows).to_string(index=False))
