"""Leg categories: what is knowable before the game about a prop UNDER, graded on what it actually paid.

Why this exists. On 2026-11-23 the Monday-night test cards missed 5 of 14 legs. Looking each one up showed
the card builder had picked NFL legs on price and prop type alone: it never read a game log, did not know one
player was in his first game back from five weeks out, and did not know which team was the underdog. This
module gives every leg those facts and, more importantly, measures on two seasons of settled Kalshi unders
whether each kind of fact has mattered, using the price a taker really paid for the under plus the fee.

A category is a fixed, pre-written description ("rushing unders on 3+ point underdogs"). Its record is compared
with the baseline for the sport (all unders), season by season:

  PREFER / AVOID   beat (trailed) the baseline by 3+ points of return in BOTH seasons, and the pooled
                   difference is at least 2 standard errors (clustered by game)
  lean prefer /    3+ points in both seasons but under 2 standard errors pooled
  lean avoid
  neutral          everything else, which is most things

Cards drop AVOID and lean-avoid legs and take PREFER legs first. None of this makes a lottery card a good bet:
the categories shade which legs go on a card, and the two PREFER categories are also handed to the fund's
pre-game walk-forward test, where they would need 3 standard errors before any single bet is placed on them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

TABLE = config.DATA_DIR / "leg_categories.json"
BANDS, SHRINK = [(0.20, 0.45), (0.45, 0.65), (0.65, 0.851)], 300.0     # price bands for a category's gap; bets of prior toward the overall gap
FEE, EDGE_PTS, Z_BAR = 0.07, 0.03, 2.0
DOG, FAV = -3.0, 3.0                      # point spread for the player's team (history)
DOG_P, FAV_P = 0.41, 0.59                 # the same thing read from Kalshi's own moneyline tonight

NFL = {  # name -> (mask function, plain description)
    "rushing unders on 3+ point underdogs": lambda d: (d.series == "KXNFLRSHYDS") & (d.team_spread <= DOG),
    "rushing unders on 3+ point favourites": lambda d: (d.series == "KXNFLRSHYDS") & (d.team_spread >= FAV),
    "tight-end reception unders": lambda d: (d.series == "KXNFLREC") & (d.position == "TE"),
    "wide-receiver reception unders": lambda d: (d.series == "KXNFLREC") & (d.position == "WR"),
    "running-back reception unders": lambda d: (d.series == "KXNFLREC") & (d.position == "RB"),
    "receiving-yards unders": lambda d: d.series == "KXNFLRECYDS",
    "passing-yards unders": lambda d: d.series == "KXNFLPASSYDS",
    "touchdown unders on 3+ point underdogs": lambda d: d.series.isin(["KXNFLTD", "KXNFLANYTD"]) & (d.team_spread <= DOG),
    "touchdown unders on 3+ point favourites": lambda d: d.series.isin(["KXNFLTD", "KXNFLANYTD"]) & (d.team_spread >= FAV),
    "first game back after 3+ weeks out": lambda d: d.weeks_out >= 3,
    "unders his recent games already support (cold streak)": lambda d: d.form_gap < -0.4,
    "unders against a hot streak": lambda d: d.form_gap > 0.4,
    "listed questionable on the injury report": lambda d: d.status == "Questionable",
    "the lead player at his position is out": lambda d: d.lead_out == 1,
    "his starting quarterback is out": lambda d: d.qb_out == 1,
}
NBA = {
    "points unders": lambda d: d.series == "KXNBAPTS",
    "rebounds unders": lambda d: d.series == "KXNBAREB",
    "assists unders": lambda d: d.series == "KXNBAAST",
    "threes unders": lambda d: d.series == "KXNBA3PT",
    "points+rebounds+assists unders": lambda d: d.series == "KXNBAPRA",
    "steals and blocks unders": lambda d: d.series.isin(["KXNBASTL", "KXNBABLK"]),
    "double-double unders": lambda d: d.series == "KXNBA2D",
    "first game back after 7+ days off": lambda d: d.days_rest >= 7,
    "second night of a back-to-back": lambda d: d.b2b == 1,
    "minutes trending down": lambda d: d.min_l3 < 0.88 * d.min_l10,
    "minutes trending up": lambda d: d.min_l3 > 1.12 * d.min_l10,
    "unders his recent games already support (cold streak)": lambda d: d.form_gap < -0.4,
    "unders against a hot streak": lambda d: d.form_gap > 0.4,
    "bench players (under 24 minutes)": lambda d: d.min_l10 < 24,
}
CATS = {"NFL": NFL, "NBA": NBA}


# ---------------------------------------------------------------- history
def _costs() -> pd.DataFrame:
    pre = pd.concat([pd.read_parquet(config.DATA_DIR / f) for f in ("prop_pre_prior.parquet", "prop_pre.parquet")
                     if (config.DATA_DIR / f).exists()])
    return pre[pre.taker_side == "no"][["ticker", "cost"]].rename(columns={"cost": "no_cost"}).drop_duplicates("ticker")


def history(sport: str) -> pd.DataFrame:
    """Every settled under with a real pre-game taker price, its features, and what it returned after the fee."""
    if sport == "NFL":
        from . import nfl_form
        d = nfl_form.dataset()
    else:
        from . import matchup
        d = matchup.dataset().rename(columns={"season_label": "season"})
    d = d.merge(_costs(), on="ticker", how="inner")
    d = d[d.no_cost.between(0.20, 0.85)].copy()
    d["c"] = d.no_cost + FEE * d.no_cost * (1 - d.no_cost)
    d["win"] = 1 - d.y
    d["net"] = d.win - d.c
    return d


def _stat(g: pd.DataFrame) -> dict:
    if len(g) < 2:
        return dict(bets=len(g), ret=float("nan"), se=float("nan"), won=float("nan"), needed=float("nan"))
    cl = g.groupby("game").net.sum()
    G = len(cl)
    se = float(np.sqrt(((cl - cl.mean()) ** 2).sum() * G / max(G - 1, 1)) / g.c.sum()) if G > 1 else float("nan")
    gap = g.win - g.no_cost                                   # how often it hit minus the price paid (before the fee)
    gc_ = gap.groupby(g.game).sum()
    gap_se = float(np.sqrt(((gc_ - gc_.mean()) ** 2).sum() * G / max(G - 1, 1)) / len(g)) if G > 1 else float("nan")
    return dict(bets=int(len(g)), ret=float(g.net.sum() / g.c.sum()), se=se, won=float(g.win.mean()), needed=float(g.c.mean()),
                gap=float(gap.mean()), gap_se=gap_se)


def label(d_last: float, d_this: float, z: float) -> str:
    both_up, both_dn = (d_last >= EDGE_PTS and d_this >= EDGE_PTS), (d_last <= -EDGE_PTS and d_this <= -EDGE_PTS)
    if both_up:
        return "PREFER" if z >= Z_BAR else "lean prefer"
    if both_dn:
        return "AVOID" if z <= -Z_BAR else "lean avoid"
    return "neutral"


def grade(sport: str, d: pd.DataFrame | None = None) -> pd.DataFrame:
    d = d if d is not None else history(sport)
    base = {s: _stat(d[d.season == s]) for s in ("last season", "this season")}
    base["both"] = _stat(d)
    rows = [dict(category=f"ALL {sport} unders (baseline)", last=base["last season"]["ret"], this=base["this season"]["ret"], both=base["both"]["ret"],
                 bets_last=base["last season"]["bets"], bets_this=base["this season"]["bets"], vs_base=0.0, z=float("nan"), label="baseline")]
    for name, f in CATS[sport].items():
        m = f(d).fillna(False)
        a, b, p = _stat(d[m & (d.season == "last season")]), _stat(d[m & (d.season == "this season")]), _stat(d[m])
        if min(a["bets"], b["bets"]) < 100:
            lab, z, dl, dt = "too few bets", float("nan"), float("nan"), float("nan")
        else:
            dl, dt = a["ret"] - base["last season"]["ret"], b["ret"] - base["this season"]["ret"]
            z = (p["ret"] - base["both"]["ret"]) / p["se"] if p["se"] and p["se"] > 0 else float("nan")
            lab = label(dl, dt, z)
        bands = []                                   # the hit-minus-price gap by price band, shrunk toward the category's overall gap
        if p.get("gap") is not None and p["bets"] >= 100:
            bands = band_gaps(d[m], p["gap"], p["gap_se"])
        rows.append(dict(category=name, last=a["ret"], this=b["ret"], both=p["ret"], bets_last=a["bets"], bets_this=b["bets"],
                         vs_base=p["ret"] - base["both"]["ret"], z=z, label=lab, gap=p.get("gap"), gap_se=p.get("gap_se"), ret_se=p["se"], gap_bands=bands))
    return pd.DataFrame(rows)


def band_gaps(sub: pd.DataFrame, gap: float, gap_se: float) -> list[list]:
    """A category's gap (win rate minus price paid) at each price level: [low, high, gap, bets, standard error].

    The band's own average is pulled toward the category's overall gap with SHRINK bets of weight, because a band has a
    third of the bets. Its standard error is never smaller than the overall one: the band's own error (the overall error
    scaled up for the smaller sample) and the overall error are combined with their overlap counted."""
    out, n_all = [], max(len(sub), 1)
    for lo, hi in BANDS:
        g_ = sub[(sub.no_cost >= lo) & (sub.no_cost < hi)]
        n = len(g_)
        gb = float((g_.win - g_.no_cost).mean()) if n else gap
        w = n / (n + SHRINK)
        se_b2 = gap_se ** 2 * n_all / max(n, 1)
        se = float(np.sqrt(w ** 2 * se_b2 + (1 - w) ** 2 * gap_se ** 2 + 2 * w * (1 - w) * (n / n_all) * se_b2)) if n else gap_se
        out.append([lo, hi, w * gb + (1 - w) * gap, int(n), max(se, gap_se)])
    return out


def save() -> dict:
    out = {"made": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    for sport in ("NBA", "NFL"):
        try:
            out[sport] = grade(sport).replace({np.nan: None}).to_dict("records")
        except Exception as e:
            out[sport] = []
            print(f"  ! {sport} categories unavailable: {e}")
    TABLE.write_text(json.dumps(out, indent=1))
    return out


def load() -> dict:
    return json.loads(TABLE.read_text()) if TABLE.exists() else {}


def labels(sport: str) -> dict:
    return {r["category"]: r for r in load().get(sport, [])}


# ---------------------------------------------------------------- tonight
def _line(title: str) -> float | None:
    import re
    m = re.search(r": (\d+)\+", title or "")
    return float(m.group(1)) if m else None


def nfl_features(c: pd.DataFrame, win_prob: dict) -> pd.DataFrame:
    """Tonight's NFL legs with the same features the history uses. `win_prob` = {team code: Kalshi's price that it wins}."""
    from . import nfl_form
    from .data import norm_name
    L = nfl_form.logs(fresh=True)
    m = c[["ticker", "series", "title", "tipoff"]].copy()
    m["name_key"] = m.title.str.split(":").str[0].map(norm_name)
    m["game_date"] = pd.to_datetime(m.tipoff).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    m["line"] = m.title.map(_line).fillna(1.0)
    m["p_mkt"] = (1 - c.mid).values                      # legs are unders, so the over's price is 1 - mid
    f = nfl_form.form_for(L, m[m.series.isin(list(nfl_form.SERIES))])
    f = f.set_index("ticker")
    out = c.copy()
    for col in ("n", "k", "avg", "weeks_out", "position", "form_gap", "last_val"):
        out[col] = out.ticker.map(f[col]) if len(f) else np.nan
    seg = out.ticker.str.split("-").str[2]
    out["kteam"] = [next((t for t in win_prob if s.startswith(t)), None) for s in seg]
    out["team_win"] = out.kteam.map(win_prob)
    out["team_spread"] = np.where(out.team_win <= DOG_P, DOG - 1, np.where(out.team_win >= FAV_P, FAV + 1, 0.0))   # only the bucket matters
    # this week's official injury report, read fresh: the player's own status, and whether his lead teammate or quarterback is out
    out["status"], out["injury"], out["lead_out"], out["qb_out"], out["inj_checked"], out["inj_posted"] = None, None, 0.0, 0.0, False, False
    try:
        pid = out.ticker.map(f["player_id"]) if len(f) else pd.Series(index=out.index, dtype=object)
        g = nfl_form._get("schedules", "games")
        g = g[["season", "week", "gameday"]].drop_duplicates()
        wk = {pd.Timestamp(r.gameday): (int(r.season), int(r.week)) for r in g.itertuples()}
        inj = nfl_form.injuries(fresh=True)
        flags = {}
        for day in m.game_date.dropna().unique():
            sw = wk.get(pd.Timestamp(day))
            if sw and sw not in flags:
                flags[sw] = (nfl_form.week_flags(L, inj, *sw).set_index("player_id"), bool(((inj.season == sw[0]) & (inj.week == sw[1])).any()))
        for i, (day, p_) in enumerate(zip(m.game_date, pid)):
            sw = wk.get(pd.Timestamp(day)) if pd.notna(day) else None
            if sw is None or sw not in flags:
                continue
            fl, published = flags[sw]
            out.iat[i, out.columns.get_loc("inj_checked")] = published
            if p_ in fl.index:
                r = fl.loc[p_]
                out.iat[i, out.columns.get_loc("status")] = r.status if isinstance(r.status, str) else None
                out.iat[i, out.columns.get_loc("injury")] = r.injury if isinstance(r.injury, str) else None
                out.iat[i, out.columns.get_loc("lead_out")], out.iat[i, out.columns.get_loc("qb_out")] = float(r.lead_out), float(r.qb_out)
                out.iat[i, out.columns.get_loc("inj_posted")] = bool(r.team_posted)
    except Exception as e:
        print(f"  (injury report unavailable: {e})")
    return out


def nba_features(c: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    from .data import KALSHI_TO_LOG_TEAM, norm_name
    from .favorites import slate_date
    out = c.copy()
    rows = []
    for r in c.itertuples():
        name = norm_name(r.title.split(":")[0])
        seg = r.ticker.split("-")[2]
        g = ctx["last"].get((name, KALSHI_TO_LOG_TEAM.get(seg[:3], seg[:3])))
        if g is None or len(g) < 5:
            rows.append((np.nan, np.nan, np.nan, 0.0, 0.0))
            continue
        day = slate_date(r.game)
        rest = (day - g.game_date.max()).days
        rows.append((rest, g.minutes.tail(3).mean(), g.minutes.tail(10).mean(), float(rest == 1), 0.0))
    for i, col in enumerate(("days_rest", "min_l3", "min_l10", "b2b", "form_gap")):
        out[col] = [x[i] for x in rows]
    return out


def classify(feat: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Adds `cats` (the graded categories each leg falls in), `pref` (2 PREFER, 1 lean prefer, 0 neutral, -1 lean avoid,
    -2 AVOID; the worst label wins) and `cat_why` (one line per category that is not neutral)."""
    lab = labels(sport)
    score = {"PREFER": 2, "lean prefer": 1, "lean avoid": -1, "AVOID": -2}
    cats, pref, why = [[] for _ in range(len(feat))], np.zeros(len(feat)), [[] for _ in range(len(feat))]
    every = [[] for _ in range(len(feat))]
    for name, f in CATS[sport].items():
        try:
            m = f(feat).fillna(False).values
        except Exception:
            continue
        for i in np.flatnonzero(m):
            every[i].append(name)
        r = lab.get(name)
        if not r or r["label"] not in score:
            continue
        for i in np.flatnonzero(m):
            cats[i].append(name)
            s = score[r["label"]]
            pref[i] = min(pref[i], s) if (s < 0 or pref[i] < 0) else max(pref[i], s)
            why[i].append(f"{r['label']}: {name} returned {r['last']:+.1%} last season and {r['this']:+.1%} this season after fees "
                          f"(all {sport} unders: {lab[f'ALL {sport} unders (baseline)']['both']:+.1%})")
    out = feat.copy()
    out["cats"], out["pref"], out["cat_why"], out["cats_all"] = cats, pref, why, every
    return out


def nfl_why(r) -> list[str]:
    """Plain facts for one NFL leg from its game log and the game line."""
    from .nfl_form import WORD
    lines = []
    if pd.notna(r.n) and r.n >= 1:
        word = WORD.get(r.series, "")
        line = _line(r.title) or 1
        lines.append(f"last {int(r.n)} games: reached {line:g}+ {word} in {int(r.k)} of them (average {r.avg:.1f}, last game {r.last_val:g}); "
                     f"the price implies {1 - r.mid:.0%} tonight")
        if pd.notna(r.weeks_out) and r.weeks_out >= 3:
            lines.append(f"first game back after about {int(r.weeks_out)} weeks out")
    else:
        lines.append("no game log this season or last: nothing to judge him on")
    if getattr(r, "inj_checked", False):
        st = getattr(r, "status", None)
        if isinstance(st, str):
            lines.append(f"injury report: {st} ({r.injury})")
        elif getattr(r, "inj_posted", False):
            lines.append("injury report: his team's game statuses are posted and he has none (expected to play)")
        else:
            lines.append("injury report: his team has not posted game statuses yet (they come out about two days before the game), so availability is not confirmed")
        if getattr(r, "lead_out", 0) == 1:
            lines.append("injury report: the lead player at his position is out or doubtful, so his share of the work should rise")
        if getattr(r, "qb_out", 0) == 1:
            lines.append("injury report: his starting quarterback is out or doubtful")
    else:
        lines.append("injury report: this week's report is not published yet")
    if pd.notna(r.team_win):
        role = "underdog" if r.team_win <= DOG_P else ("favourite" if r.team_win >= FAV_P else "close game")
        lines.append(f"{r.kteam} is priced {r.team_win:.0%} to win ({role})")
    return lines + list(r.cat_why)


def run(args=None) -> None:
    pd.set_option("display.width", 230)
    out = save()
    for sport in ("NFL", "NBA"):
        t = pd.DataFrame(out[sport])
        if t.empty:
            continue
        for col in ("last", "this", "both", "vs_base"):
            t[col] = t[col].map(lambda v: "" if v is None or v != v else f"{v:+.1%}")
        t["z"] = t.z.map(lambda v: "" if v is None or v != v else f"{v:+.1f}")
        t = t.drop(columns=[c_ for c_ in ("gap", "gap_se", "ret_se", "gap_bands") if c_ in t])
        print(f"\n{sport} UNDERS by what was knowable before the game (return per $1 after the fee, real taker prices)")
        print(t.rename(columns={"last": "last_season", "this": "this_season", "both": "both_seasons", "vs_base": "vs_baseline"}).to_string(index=False))
    print("\nPREFER/AVOID = 3+ points better/worse than the baseline in BOTH seasons and 2+ standard errors pooled. 'lean' = both seasons, under 2 standard errors.")
    print("These shade which legs go on a lottery card. A single bet needs 3 standard errors in the walk-forward test (`python -m bot walk --pre`).")
