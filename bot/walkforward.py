"""Walk-forward backtest of the fund's LOGIC (not of one hand-picked rule).

On every night in the history the bot only knows what happened on earlier nights. From a fixed
family of ~150 candidate rules it must decide, by itself, which have earned the right to be bet:

    a rule is ACTIVE tonight if, on earlier nights only, it has at least `min_bets` bets over
    `min_games` games and its return is positive even `z` standard errors below its estimate.

Active rules are staked at a fraction of the Kelly bet implied by that lower bound, capped by how
many dollars can realistically be filled per bet and by total exposure. Then the night's real
outcomes are applied and the record is updated. Nothing from the future is ever used, so the
return earned AFTER a rule is activated is genuinely out-of-sample - including the rule we would
have picked with hindsight, and including every rule that looked good for a while and then failed.

Candidate family (all "buy once, first in-game fill" rules, from data already on disk):
    player props   prop type x OVER/UNDER x price band (35-50c, 50-65c, 65-80c, 80-95c)
    moneylines     NBA and NFL, first touch of 90c / 95c / 97c
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .trades import TAKER_FEE


def load_bets() -> pd.DataFrame:
    """Every candidate bet in both seasons: rule, date, entry/resolve time, cost, win, game, ticker."""
    from .favorites import slate_date
    from .props_study import STAT
    from .trades import PRIOR_TOUCH
    parts = []
    for f in ("prop_touch_prior.parquet", "prop_touch.parquet"):
        fp = config.DATA_DIR / f
        if not fp.exists():
            continue
        t = pd.read_parquet(fp)
        t["rule"] = t.series.map(STAT) + " " + np.where(t.taker_side == "yes", "OVER", "UNDER") + " " + t.band
        t["win"] = (t.taker_side == t.result).astype(float)
        t["unit"] = t.ticker + ":" + t.taker_side                 # one position per market side
        parts.append(t[["rule", "ticker", "unit", "game", "created_time", "mins", "cost", "win"]])
    for fp in (PRIOR_TOUCH, config.DATA_DIR / "first_touch.parquet"):
        if not fp.exists():
            continue
        t = pd.read_parquet(fp)
        t = t[t.series.isin(["KXNBAGAME", "KXNFLGAME"]) & t.threshold.isin([0.90, 0.95, 0.97])]
        t = t.sort_values("created_time").groupby(["league", "threshold", "game"], as_index=False).first()
        t["rule"] = t.league + " moneyline first touch " + (t.threshold * 100).round().astype(int).astype(str) + "c"
        t["unit"] = t.game + ":ml"
        parts.append(t[["rule", "ticker", "unit", "game", "created_time", "mins", "cost", "win"]])
    b = pd.concat(parts, ignore_index=True)
    b["date"] = b.game.map(slate_date)
    b["resolve"] = b.created_time + pd.to_timedelta((150 - b.mins).clip(lower=5), unit="m")
    b["c"] = b.cost + TAKER_FEE * b.cost * (1 - b.cost)            # cost per contract with fee
    b["net"] = b.win - b.c
    return b.dropna(subset=["date"]).sort_values(["date", "created_time"]).reset_index(drop=True)


def _records(b: pd.DataFrame) -> pd.DataFrame:
    """Per rule and night: cumulative record over EARLIER nights only (bets, games, return, std error)."""
    g = b.groupby(["rule", "date", "game"], as_index=False).agg(net=("net", "sum"), c=("c", "sum"), n=("net", "size"))
    g["net2"] = g.net ** 2
    d = g.groupby(["rule", "date"], as_index=False).agg(S1=("net", "sum"), S2=("net2", "sum"), G=("game", "size"),
                                                         C=("c", "sum"), N=("n", "sum"))
    d = d.sort_values(["rule", "date"])
    for col in ("S1", "S2", "G", "C", "N"):
        d["p" + col] = d.groupby("rule")[col].cumsum() - d[col]    # prior nights only
    G = d.pG.clip(lower=2)
    d["roi"] = d.pS1 / d.pC.replace(0, np.nan)
    d["se"] = np.sqrt(((d.pS2 - d.pS1 ** 2 / G) * G / (G - 1)).clip(lower=0)) / d.pC.replace(0, np.nan)
    d["cbar"] = d.pC / d.pN.replace(0, np.nan)
    # recent form: the rule's last `w` nights with bets, earlier nights only
    for w in (20, 40):
        r1 = d.groupby("rule").S1.transform(lambda x: x.shift(1).rolling(w, min_periods=max(5, w // 4)).sum())
        rc = d.groupby("rule").C.transform(lambda x: x.shift(1).rolling(w, min_periods=max(5, w // 4)).sum())
        d[f"recent{w}"] = r1 / rc
    return d


_REC_CACHE = {}


def run(z: float = 3.0, min_bets: int = 150, min_games: int = 60, kelly_mult: float = 0.25, max_stake: float = 5.0,
        max_concurrent: float = 0.5, bankroll: float = 100.0, bets: pd.DataFrame | None = None, verbose: bool = False,
        recent: int = 0, recent_min: float = 0.0) -> dict:
    """recent=20 or 40: the rule must ALSO have returned more than `recent_min` over its last 20/40 nights
    with bets (earlier nights only) - a check that it is still working, not just that it once did."""
    b = bets if bets is not None else load_bets()
    if id(b) not in _REC_CACHE:
        _REC_CACHE.clear()
        _REC_CACHE[id(b)] = _records(b).set_index(["rule", "date"])
    rec = _REC_CACHE[id(b)].copy()
    rec["lb"] = rec.roi - z * rec.se
    rec["active"] = (rec.pN >= min_bets) & (rec.pG >= min_games) & (rec.lb > 0)
    if recent:
        rec["active"] &= rec[f"recent{recent}"] > recent_min
    rec["frac"] = (kelly_mult * rec.cbar * rec.lb / (1 - rec.cbar)).clip(lower=0, upper=0.25).where(rec.active, 0.0)
    bal, start, path, log = bankroll, bankroll, [], []
    first_active = {}
    live = b.join(rec[["active", "frac"]], on=["rule", "date"])
    live = live[live.active.fillna(False)]
    all_dates = sorted(b.date.unique())
    by_night = {d: g for d, g in live.groupby("date")}
    for date in all_dates:
        night = by_night.get(date)
        if night is None:
            path.append((date, bal))
            continue
        cash, open_pos, taken = bal, [], set()
        for r in night.itertuples():
            key = (r.rule, date)
            if r.unit in taken:
                continue
            first_active.setdefault(r.rule, date)
            for pos in [p for p in open_pos if p[0] <= r.created_time]:
                cash += pos[1]
                open_pos.remove(pos)
            exposure = sum(p[2] for p in open_pos)
            stake = min(r.frac * bal, cash, max(max_concurrent * bal - exposure, 0.0), max_stake)
            if stake < 0.01:
                continue
            taken.add(r.unit)
            cash -= stake
            open_pos.append((r.resolve, stake / r.c if r.win else 0.0, stake))
            log.append((date, r.rule, r.ticker, stake, stake * (1 / r.c - 1) if r.win else -stake, r.win))
        bal = cash + sum(p[1] for p in open_pos)
        path.append((date, bal))
    L = pd.DataFrame(log, columns=["date", "rule", "ticker", "stake", "pnl", "win"])
    P = pd.DataFrame(path, columns=["date", "bankroll"]).set_index("date")
    peak = P.bankroll.cummax()
    return dict(final=bal, bets=len(L), staked=L.stake.sum() if len(L) else 0.0, pnl=L.pnl.sum() if len(L) else 0.0,
                roi=(L.pnl.sum() / L.stake.sum()) if len(L) and L.stake.sum() else float("nan"),
                max_drawdown=float((peak - P.bankroll).max()), worst_night=float(P.bankroll.diff().min()),
                losing_nights=float((P.bankroll.diff() < 0).mean()), log=L, path=P, first_active=first_active, rec=rec, bets_all=b)


def report(max_stake: float = 5.0):
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    b = load_bets()
    print(f"\nWALK-FORWARD BACKTEST OF THE FUND'S LOGIC on data already downloaded: {len(b):,} candidate bets, "
          f"{b.rule.nunique()} rules, {b.date.nunique()} nights ({b.date.min():%b %Y} - {b.date.max():%b %Y}).")
    print("Each night the bot sees only earlier nights, activates the rules whose past record clears the bar, "
          f"stakes a quarter of the cautious Kelly bet (at most ${max_stake:g} per bet), and takes the real result.\n")

    print("1) HOW STRICT SHOULD THE BAR BE?  (start $100)")
    rows = []
    for z in (0.0, 1.64, 2.5, 3.0, 4.0):
        r = run(z=z, max_stake=max_stake, bets=b)
        rows.append(dict(bar=f"return > 0 by {z:g} std errors" if z else "any positive record (no bar)",
                         rules_ever_used=len(r["first_active"]), bets=r["bets"], staked=f"${r['staked']:,.0f}",
                         profit=f"${r['pnl']:+,.0f}", return_on_stakes=f"{r['roi']:+.1%}", end=f"${r['final']:,.0f}",
                         worst_night=f"${r['worst_night']:,.0f}", max_drawdown=f"${r['max_drawdown']:,.0f}",
                         losing_nights=f"{r['losing_nights']:.0%}"))
    print(pd.DataFrame(rows).to_string(index=False))

    r = run(z=3.0, max_stake=max_stake, bets=b)
    L = r["log"]
    print("\n2) WHAT THE LOGIC ACTUALLY BET (bar = 3 std errors), rule by rule - everything here is AFTER the rule was activated,"
          " so it is out-of-sample:")
    if len(L):
        t = L.groupby("rule").agg(first_bet=("date", "min"), bets=("pnl", "size"), won=("win", "mean"), staked=("stake", "sum"),
                                  profit=("pnl", "sum")).reset_index()
        t["return"] = t.profit / t.staked
        t = t.sort_values("profit", ascending=False)
        f = t.copy()
        f["first_bet"] = f.first_bet.dt.strftime("%Y-%m-%d")
        f["won"] = f.won.map("{:.1%}".format); f["staked"] = f.staked.map("${:,.0f}".format)
        f["profit"] = f.profit.map("${:+,.0f}".format); f["return"] = f["return"].map("{:+.1%}".format)
        print(f.to_string(index=False))
        print(f"   {int((t.profit > 0).sum())} of {len(t)} activated rules made money after activation.")
    else:
        print("   no rule ever cleared the bar")

    print("\n3) SEASON BY SEASON (bar = 3 std errors):")
    if len(L):
        L = L.assign(season=np.where(L.date < pd.Timestamp("2026-07-01"), "last season", "this season"))
        s = L.groupby("season").agg(bets=("pnl", "size"), won=("win", "mean"), staked=("stake", "sum"), profit=("pnl", "sum"))
        s["return"] = s.profit / s.staked
        print(s.round(3).to_string())

    print("\n3b) DOES A RECENCY CHECK FIX THIS SEASON?  (bar = 3 std errors; rule must also be profitable over its recent nights)")
    rows = []
    for name, kw in (("lifetime record only (the logic so far)", {}),
                     ("+ profitable over its last 40 nights", dict(recent=40)),
                     ("+ profitable over its last 20 nights", dict(recent=20)),
                     ("+ returned over 3% in its last 20 nights", dict(recent=20, recent_min=0.03))):
        x = run(z=3.0, max_stake=max_stake, bets=b, **kw)
        Lx = x["log"]
        row = dict(logic=name, rules=len(x["first_active"]), bets=x["bets"])
        for lab, m in (("last_season", Lx.date < pd.Timestamp("2026-07-01")), ("this_season", Lx.date >= pd.Timestamp("2026-07-01"))):
            d = Lx[m]
            row[lab] = f"{d.pnl.sum() / d.stake.sum():+.1%} (${d.pnl.sum():+,.0f})" if len(d) and d.stake.sum() else "no bets"
        row |= dict(end=f"${x['final']:,.0f}", max_drawdown=f"${x['max_drawdown']:,.0f}")
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n3c) THIS SEASON ONLY, rule by rule (lifetime-record logic):")
    if len(L):
        ts = L[L.date >= pd.Timestamp("2026-07-01")].groupby("rule").agg(bets=("pnl", "size"), won=("win", "mean"),
                                                                        staked=("stake", "sum"), profit=("pnl", "sum"))
        ts["return"] = ts.profit / ts.staked
        print(ts.sort_values("profit").round(3).to_string())

    print("\n4) HOW MUCH CAN BE FILLED PER BET?  (bar = 3 std errors)")
    rows = []
    for cap in (1, 2, 5, 20, float("inf")):
        x = run(z=3.0, max_stake=cap, bets=b)
        rows.append(dict(fillable_per_bet="no limit (unrealistic)" if cap == float("inf") else f"${cap:g}", bets=x["bets"],
                         staked=f"${x['staked']:,.0f}", profit=f"${x['pnl']:+,.0f}", end=f"${x['final']:,.0f}",
                         max_drawdown=f"${x['max_drawdown']:,.0f}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n5) THE HINDSIGHT TRAP: rules that looked good early and what they did next (bar = 1.64 std errors, the looser bar)")
    r2 = run(z=1.64, max_stake=max_stake, bets=b)
    L2 = r2["log"]
    if len(L2):
        t2 = L2.groupby("rule").agg(first_bet=("date", "min"), bets=("pnl", "size"), profit=("pnl", "sum"), staked=("stake", "sum"))
        t2["return"] = t2.profit / t2.staked
        bad = t2[t2.profit < 0].sort_values("profit").head(8).reset_index()
        bad["first_bet"] = bad.first_bet.dt.strftime("%Y-%m-%d")
        print(f"   {len(t2)} rules were activated at the looser bar; {int((t2.profit < 0).sum())} lost money afterwards. The worst:")
        print(bad.round(3).to_string(index=False))
    return r


def active_now(z: float = 3.0, min_bets: int = 150, min_games: int = 60, bets: pd.DataFrame | None = None) -> pd.DataFrame:
    """What the logic would bet on the NEXT night: every rule's full record to date, with the same test."""
    b = bets if bets is not None else load_bets()
    g = b.groupby(["rule", "game"], as_index=False).agg(net=("net", "sum"), c=("c", "sum"), n=("net", "size"), date=("date", "max"))
    rows = []
    for rule, d in g.groupby("rule"):
        G, C, S1 = len(d), d.c.sum(), d.net.sum()
        if G < 2:
            continue
        se = math.sqrt(max((d.net ** 2).sum() - S1 ** 2 / G, 0) * G / (G - 1)) / C
        nights = d.groupby("date").agg(net=("net", "sum"), c=("c", "sum")).sort_index()
        r20 = nights.net.tail(20).sum() / nights.c.tail(20).sum()
        this = d[d.date >= pd.Timestamp("2026-07-01")]
        rows.append(dict(rule=rule, bets=int(d.n.sum()), games=G, ret=S1 / C, se=se, low_end=S1 / C - z * se,
                         last_20_nights=r20, this_season_bets=int(this.n.sum()),
                         this_season=this.net.sum() / this.c.sum() if len(this) else float("nan"),
                         last_bet=d.date.max()))
    t = pd.DataFrame(rows)
    t["passes_lifetime"] = (t.bets >= min_bets) & (t.games >= min_games) & (t.low_end > 0)
    t["passes_recent"] = t.passes_lifetime & (t.last_20_nights > 0.03)
    return t.sort_values("low_end", ascending=False)
