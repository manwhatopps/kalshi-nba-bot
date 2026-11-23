"""Scenarios: does THIS kind of player have an advantage against THIS kind of defense?

The idea being tested is the bettor's one: a pull-up three-point shooter facing a drop-coverage defense
should get more and easier pull-up looks, so his points and threes should go over. Earlier studies looked
at the defense alone, or at broad player roles. This one uses each player's own shot diet.

Data: github.com/gabriel1200/player_sheets, player-level tracking for every game of 2025-26 and 2026-27
(pull-up threes, catch-and-shoot threes, shots by zone, drives, post touches, potential assists, rebound
chances). No licence file, so it is downloaded at run time for private analysis and never committed.

Everything is walk-forward: a game only sees earlier games.

  shot diet        each player's rate per minute of every shot type over his previous 15 games
  what a defense   for every game, what its opponents actually took of each shot type divided by what
  concedes         those same players normally take in those minutes; averaged over its previous 15 games.
                   1.15 = this defense gives up 15% more of that shot than its opponents usually get.
                   Opponent-adjusted, so playing Golden State does not make a defense look like it
                   "allows threes".
  exposure         a player's diet weighted by what tonight's defense concedes: the percentage more (or
                   fewer) looks he should get if the defense treats him the way it has treated everyone.

Two questions are asked of every scenario, and they are different questions:
  1. Is the advantage REAL?  Did such players actually produce more than their own norm in such games?
  2. Is it PRICED?           Did the over hit more often than Kalshi's price said?
An advantage that is real but priced is a reason the line is where it is, not a reason to bet.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import config
from .data import norm_name

EXT = config.DATA_DIR / "ext"
URL = "https://raw.githubusercontent.com/gabriel1200/player_sheets/master/game_report/all_games/all_{y}.parquet"
SEASON = {2026: "last season", 2027: "this season"}
# shot types (a partition of field-goal attempts: twos by zone, threes by how they are taken)
SHOTS = {"rim": "shots at the rim", "paint": "floaters and short paint shots", "mid": "mid-range jumpers",
         "pu3": "pull-up threes", "cs3": "catch-and-shoot threes", "o3": "other threes"}
OTHER = {"pu2": "pull-up twos", "drives": "drives", "post": "post touches", "pot_ast": "potential assists",
         "reb_ch": "rebound chances", "fta": "free throws", "fga": "shots", "pts": "points", "fg3a": "three-point attempts",
         "fg3m": "threes made", "ast": "assists", "reb": "rebounds"}
TYPES = list(SHOTS) + list(OTHER)
WINDOW, MIN_GAMES = 15, 5


# ---------------------------------------------------------------- data
def load(years=(2026, 2027), max_age_h: float = 12) -> pd.DataFrame:
    EXT.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years:
        f = EXT / f"player_games_{y}.parquet"
        if not f.exists() or (y == max(years) and (time.time() - f.stat().st_mtime) / 3600 > max_age_h):
            try:
                r = requests.get(URL.format(y=y), timeout=300)
                r.raise_for_status()
                f.write_bytes(r.content)
            except Exception as e:
                if not f.exists():
                    print(f"  ! player tracking for {y} unavailable: {e}")
                    continue
        d = pd.read_parquet(f)
        d = d[~d.playoffs.astype(bool)]
        z = lambda c: d[c].fillna(0.0).astype(float)
        pu3, cs3 = z("pullup_PULL_UP_FG3A"), z("CATCH_SHOOT_FG3A")
        parts.append(pd.DataFrame({
            "player_id": d.PLAYER_ID, "name": d.PLAYER_NAME, "name_key": d.PLAYER_NAME.map(norm_name),
            "team": d.TEAM_ABBREVIATION, "opp": d.opp_team_abbr, "game_id": d.GAME_ID,
            "game_date": pd.to_datetime(d["date"].astype(str), format="%Y%m%d"), "season": SEASON.get(y, str(y)),
            "min": z("MIN"), "rim": z("RA_FGA"), "paint": z("ITP_FGA"), "mid": z("MID_FGA"), "pu3": pu3, "cs3": cs3,
            "o3": (z("FG3A") - pu3 - cs3).clip(lower=0), "pu2": (z("pullup_PULL_UP_FGA") - pu3).clip(lower=0),
            "drives": z("DRIVES"), "post": z("POST_TOUCHES"), "pot_ast": z("POTENTIAL_AST"), "reb_ch": z("REB_CHANCES"),
            "fta": z("FTA"), "fga": z("FGA"), "pts": z("PTS"), "fg3a": z("FG3A"), "fg3m": z("FG3M"), "ast": z("AST"), "reb": z("REB")}))
    if not parts:
        return pd.DataFrame()
    p = pd.concat(parts, ignore_index=True)
    return p[p["min"] > 0].sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


def profiles(p: pd.DataFrame) -> pd.DataFrame:
    """Adds r_<type>: the player's per-minute rate over his previous WINDOW games (NaN until MIN_GAMES)."""
    g = p.groupby("player_id")
    mins = g["min"].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).sum())
    out = {"min_l": g["min"].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())}
    for t in TYPES:
        out["r_" + t] = g[t].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).sum()) / mins
    return pd.concat([p, pd.DataFrame(out, index=p.index)], axis=1)


def concedes(p: pd.DataFrame) -> pd.DataFrame:
    """Per defense and game: c_<type> = (what opponents took) / (what those players normally take in those
    minutes), over the defense's PREVIOUS WINDOW games."""
    q = p.dropna(subset=["r_fga"])
    cols = {}
    for t in TYPES:
        cols["a_" + t] = q[t]
        cols["e_" + t] = q["r_" + t] * q["min"]
    x = pd.concat([q[["opp", "game_id", "game_date"]], pd.DataFrame(cols, index=q.index)], axis=1)
    tg = x.groupby(["opp", "game_id", "game_date"], as_index=False).sum().sort_values(["opp", "game_date"])
    g = tg.groupby("opp")
    out = tg[["opp", "game_id", "game_date"]].copy()
    for t in TYPES:
        a = g["a_" + t].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).sum())
        e = g["e_" + t].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).sum())
        out["c_" + t] = a / e.where(e > 0)
    return out.rename(columns={"opp": "defense"})


def add_exposure(d: pd.DataFrame) -> pd.DataFrame:
    """Shot-diet exposure for each player-game row that has both a profile and the defense's record."""
    tot = sum(d["r_" + s] for s in SHOTS).where(lambda v: v > 0)
    three = (d.r_pu3 + d.r_cs3 + d.r_o3).where(lambda v: v > 0)
    d = d.copy()
    d["x_pts"] = sum((d["r_" + s] / tot) * (d["c_" + s] - 1) for s in SHOTS)                  # expected change in shots, from diet
    d["x_3pt"] = sum((d["r_" + s] / three) * (d["c_" + s] - 1) for s in ("pu3", "cs3", "o3"))
    d["x_ast"], d["x_reb"] = d.c_pot_ast - 1, d.c_reb_ch - 1
    d["sh_pu3"] = d.r_pu3 / tot                                                                # share of his shots that are pull-up threes
    d["sh_cs3"], d["sh_rim"], d["sh_mid"] = d.r_cs3 / tot, d.r_rim / tot, (d.r_mid + d.r_paint) / tot
    d["pu3_36"], d["cs3_36"], d["drv_36"], d["post_36"], d["past_36"] = (36 * d[c] for c in ("r_pu3", "r_cs3", "r_drives", "r_post", "r_pot_ast"))
    # drop coverage sits the big at the rim: pull-up jumpers conceded, rim and kick-out threes taken away
    d["drop"] = (d.c_pu3 + d.c_pu2) / 2 - (d.c_rim + d.c_cs3) / 2
    return d


def build(p: pd.DataFrame | None = None) -> pd.DataFrame:
    p = profiles(p if p is not None else load())
    c = concedes(p)
    d = p.merge(c.drop(columns="game_date"), left_on=["opp", "game_id"], right_on=["defense", "game_id"], how="left")
    return add_exposure(d)


# ---------------------------------------------------------------- the named scenarios
def player_types(d: pd.DataFrame) -> dict:
    """Who counts as what, from the previous 15 games (fixed thresholds, per 36 minutes)."""
    return {
        "pull-up three shooter": (d.pu3_36 >= 2.5) & (d.min_l >= 20),
        "spot-up shooter": (d.cs3_36 >= 4.0) & (d.sh_pu3 < 0.08) & (d.min_l >= 18),
        "driver": (d.drv_36 >= 11) & (d.min_l >= 20),
        "rim finisher": (d.sh_rim >= 0.45) & (d.min_l >= 18),
        "post scorer": (d.post_36 >= 4.0) & (d.min_l >= 18),
        "playmaker": (d.past_36 >= 11) & (d.min_l >= 22)}


# (id, player type, defense measure, what a HIGH value of it means, box-score stats to check, Kalshi series it should move)
SCEN = [
    ("pullup_drop", "pull-up three shooter", "drop", "drop-style defense (concedes pull-up jumpers, protects the rim and stays home on shooters)",
     ("pts", "fg3m", "pu3"), ("KXNBAPTS", "KXNBA3PT")),
    ("pullup_pu3", "pull-up three shooter", "c_pu3", "defense that concedes pull-up threes", ("pts", "fg3m", "pu3"), ("KXNBAPTS", "KXNBA3PT")),
    ("spotup_cs3", "spot-up shooter", "c_cs3", "defense that concedes catch-and-shoot threes (helps off shooters)", ("pts", "fg3m", "cs3"), ("KXNBAPTS", "KXNBA3PT")),
    ("driver_lane", "driver", "c_drives", "defense that concedes drives", ("pts", "drives", "fta"), ("KXNBAPTS",)),
    ("rim_soft", "rim finisher", "c_rim", "defense that concedes shots at the rim", ("pts", "rim"), ("KXNBAPTS",)),
    ("post_allowed", "post scorer", "c_post", "defense that concedes post touches", ("pts", "post"), ("KXNBAPTS",)),
    ("maker_passing", "playmaker", "c_pot_ast", "defense that concedes assisted looks (the opposite end: switching, isolation-forcing)",
     ("ast", "pot_ast"), ("KXNBAAST",)),
]


def _rel(d: pd.DataFrame, stat: str) -> pd.Series:
    """This game's per-minute production of `stat` relative to the player's own norm (0 = as usual)."""
    exp = d["r_" + stat] * d["min"]
    return ((d[stat] - exp) / exp.where(exp > 0)).replace([np.inf, -np.inf], np.nan)


def _mean_se(x: pd.Series, groups: pd.Series) -> tuple[float, float, int]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return float("nan"), float("nan"), len(x)
    g = groups.loc[x.index]
    s = (x - x.mean()).groupby(g).sum()
    return float(x.mean()), float(np.sqrt((s ** 2).sum()) / len(x)), len(x)


def _split(x: pd.Series, hi: pd.Series, lo: pd.Series, groups: pd.Series) -> dict:
    """Mean of x where the defense is in the top third vs the bottom third of the measure, and the difference."""
    a, sa, na = _mean_se(x[hi], groups)
    b, sb, nb = _mean_se(x[lo], groups)
    se = np.sqrt(sa ** 2 + sb ** 2) if sa == sa and sb == sb else float("nan")
    return dict(hi=a, lo=b, diff=a - b, z=(a - b) / se if se and se > 0 else float("nan"), n_hi=na, n_lo=nb)


def _thirds(d: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    q = d.groupby("season")[col]
    return d[col] > q.transform(lambda s: s.quantile(2 / 3)), d[col] < q.transform(lambda s: s.quantile(1 / 3))


def attach_markets(d: pd.DataFrame) -> pd.DataFrame:
    """Settled Kalshi prop markets joined to that player's row for that game."""
    from .matchup import markets
    m = markets().drop(columns=["team", "opp"], errors="ignore")
    cols = ["name_key", "game_date", "season", "min_l", "drop"] + [c for c in d.columns if c[:2] in ("x_", "c_", "r_", "sh")] + \
           ["pu3_36", "cs3_36", "drv_36", "post_36", "past_36"]
    j = m.merge(d[cols].drop_duplicates(["name_key", "game_date"]), on=["name_key", "game_date"], how="inner")
    return j[(j.p_mkt > 0.08) & (j.p_mkt < 0.92)]


def study(verbose: bool = True) -> dict:
    """Both questions for every scenario, both seasons. The comparison is always the SAME kind of player against
    the top third of defenses on the measure vs the bottom third, so a player type's own quirks cancel out."""
    pd.set_option("display.width", 250)
    d = build()
    d = d.dropna(subset=["r_fga", "c_fga"])
    d = d[d["min"] >= 12].reset_index(drop=True)
    m = attach_markets(d).reset_index(drop=True)
    td, tm = player_types(d), player_types(m)
    rows = []
    for sid, ptype, col, meaning, stats, series in SCEN:
        for season in ("last season", "this season", "both"):
            g = d[td[ptype] & ((d.season == season) | (season == "both"))]
            hi, lo = _thirds(g, col)
            row = dict(scenario=sid, player=ptype, defense=meaning, season=season, games_hi=int(hi.sum()), games_lo=int(lo.sum()),
                       def_pts_hi=float(g.c_pts[hi].mean()), def_pts_lo=float(g.c_pts[lo].mean()))
            for s_ in stats:
                r = _split(_rel(g, s_), hi, lo, g.game_id)
                row[s_ + "_hi"], row[s_ + "_lo"], row[s_ + "_diff"], row[s_ + "_z"] = r["hi"], r["lo"], r["diff"], r["z"]
            mm = m[tm[ptype] & m.series.isin(series) & ((m.season == season) | (season == "both"))]
            mh, ml = _thirds(mm, col)
            r = _split(mm.y - mm.p_mkt, mh, ml, mm.game)
            row.update(mk_hi=r["hi"], mk_lo=r["lo"], mk_diff=r["diff"], mk_z=r["z"], markets_hi=r["n_hi"], markets_lo=r["n_lo"])
            rows.append(row)
    tab = pd.DataFrame(rows)

    # the same idea for every player at once: shot-diet exposure vs own-norm production, and vs the market after the price
    cont = []
    for stat, x, series in (("pts", "x_pts", "KXNBAPTS"), ("fg3m", "x_3pt", "KXNBA3PT"), ("ast", "x_ast", "KXNBAAST"), ("reb", "x_reb", "KXNBAREB")):
        for season in ("last season", "this season"):
            g = d[(d.season == season) & (d.min_l >= 18)].dropna(subset=[x])
            hi, lo = _thirds(g, x)
            r = _split(_rel(g, stat), hi, lo, g.game_id)
            mm = m[(m.season == season) & (m.series == series)].dropna(subset=[x])
            coef = zc = float("nan")
            if len(mm) > 500:
                import statsmodels.api as sm
                X = pd.DataFrame({"price": np.log(mm.p_mkt / (1 - mm.p_mkt)), "x": (mm[x] - mm[x].mean()) / mm[x].std()})
                f = sm.Logit(mm.y.values, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(mm.game)[0]})
                coef, zc = float(f.params["x"]), float(f.tvalues["x"])
            cont.append(dict(stat=OTHER[stat], season=season, player_games=len(g), favourable_third=r["hi"], unfavourable_third=r["lo"],
                             diff=r["diff"], diff_z=r["z"], markets=len(mm), beyond_price=coef, beyond_price_z=zc))
    cont = pd.DataFrame(cont)
    if verbose:
        _print(tab, cont, d, m)
    return dict(table=tab, cont=cont, data=d, markets=m)


def verdict(diff: float, z: float) -> str:
    if not (z == z):
        return "too little data"
    if abs(z) < 2:
        return "no measurable difference"
    return ("ADVANTAGE" if diff > 0 else "DISADVANTAGE") + (" (strong)" if abs(z) >= 3 else "")


def _print(tab: pd.DataFrame, cont: pd.DataFrame, d: pd.DataFrame, m: pd.DataFrame) -> None:
    print(f"\nSCENARIOS: player type against defensive style. {len(d):,} player-games with tracking data, {len(m):,} settled Kalshi prop markets.")
    print("  Each block compares the SAME kind of player against the third of defenses highest on a measure vs the third lowest.")
    print("  REAL?   production per minute against the player's own previous-15-game norm; 'diff' is high-third minus low-third.")
    print("  PRICED? how often the OVER beat Kalshi's pre-game price (hit rate - price) in each third; a real edge shows as a diff with |z| > 2 in BOTH seasons.\n")
    for sid, ptype, col, meaning, stats, series in SCEN:
        t = tab[tab.scenario == sid]
        print(f"  {ptype.upper()} vs a {meaning}")
        for r in t.itertuples():
            real = "   ".join(f"{OTHER.get(s, SHOTS.get(s, s))}: {getattr(r, s + '_hi'):+.1%} vs {getattr(r, s + '_lo'):+.1%}, diff {getattr(r, s + '_diff'):+.1%} (z {getattr(r, s + '_z'):+.1f})"
                              for s in stats)
            print(f"     {r.season:11s} {r.games_hi:4d}/{r.games_lo:<4d} games | REAL   {real}")
            print(f"     {'':11s} {'':9s}       | PRICED over vs price: {r.mk_hi:+.1%} vs {r.mk_lo:+.1%}, diff {r.mk_diff:+.1%} (z {r.mk_z:+.1f}; {r.markets_hi}/{r.markets_lo} markets)"
                  f"   [those defenses concede {r.def_pts_hi - 1:+.1%} / {r.def_pts_lo - 1:+.1%} points overall]")
        b = t[t.season == "both"].iloc[0]
        main = stats[0]
        print(f"     -> for {OTHER[main]}: {verdict(getattr(b, main + '_diff'), getattr(b, main + '_z'))}; "
              f"against the price: {'no measurable difference (priced)' if not abs(b.mk_z) >= 2 else ('overs did BETTER' if b.mk_diff > 0 else 'overs did WORSE') + ' (check both seasons above)'}\n")
    print("  The same idea for every player at once: shot-diet exposure (the favourable third of matchups vs the unfavourable third).")
    c = cont.copy()
    for col in ("favourable_third", "unfavourable_third", "diff"):
        c[col] = c[col].map(lambda v: f"{v:+.1%}")
    for col in ("diff_z", "beyond_price_z"):
        c[col] = c[col].map(lambda v: f"{v:+.1f}")
    c["beyond_price"] = c.beyond_price.map(lambda v: f"{v:+.3f}")
    print(c.to_string(index=False))
    print("  diff = how much more the favourable third produced than the unfavourable third, per minute, against their own norms.")
    print("  beyond_price = what one standard deviation of exposure adds to the odds of the OVER after Kalshi's price is accounted for.")


# ---------------------------------------------------------------- tonight: the scenario behind one leg
TABLE = config.DATA_DIR / "scenario_table.json"
SERIES_STAT = {"KXNBAPTS": ("pts", "x_pts"), "KXNBAPRA": ("pts", "x_pts"), "KXNBA3PT": ("fg3m", "x_3pt"), "KXNBAAST": ("ast", "x_ast"),
               "KXNBAREB": ("reb", "x_reb"), "KXNBA2D": ("reb", "x_reb")}


def save_table(res: dict) -> dict:
    """The study's verdicts in a small file, so tonight's legs can quote them without re-running it."""
    import json
    tab, cont, d = res["table"], res["cont"], res["data"]
    out = {"made": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "scenarios": {}, "exposure": {}}
    for sid, ptype, col, meaning, stats, series in SCEN:
        t = tab[tab.scenario == sid].set_index("season")
        main, b = stats[0], t.loc["both"]
        same = bool(np.sign(t.loc["last season", main + "_diff"]) == np.sign(t.loc["this season", main + "_diff"]))
        per = {s_: dict(diff=float(b[s_ + "_diff"]), z=float(b[s_ + "_z"]),
                        same_sign=bool(np.sign(t.loc["last season", s_ + "_diff"]) == np.sign(t.loc["this season", s_ + "_diff"]))) for s_ in stats}
        out["scenarios"][sid] = dict(player=ptype, defense=meaning, col=col, stat=main, series=list(series), real_diff=float(b[main + "_diff"]),
                                     real_z=float(b[main + "_z"]), same_sign=same, mk_diff=float(b.mk_diff), mk_z=float(b.mk_z), stats=per,
                                     def_pts_hi=float(b.def_pts_hi), def_pts_lo=float(b.def_pts_lo),
                                     games=int(b.games_hi + b.games_lo), markets=int(b.markets_hi + b.markets_lo))
    for stat, x in (("pts", "x_pts"), ("fg3m", "x_3pt"), ("ast", "x_ast"), ("reb", "x_reb")):
        c = cont[cont.stat == OTHER[stat]].set_index("season")
        a, b = c.loc["last season"], c.loc["this season"]
        out["exposure"][x] = dict(stat=stat, q33=float(d[x].quantile(1 / 3)), q67=float(d[x].quantile(2 / 3)),
                                  diff_last=float(a["diff"]), z_last=float(a.diff_z), diff_this=float(b["diff"]), z_this=float(b.diff_z),
                                  price_z_last=float(a.beyond_price_z), price_z_this=float(b.beyond_price_z))
    TABLE.write_text(json.dumps(out, indent=1))
    return out


def load_table() -> dict:
    import json
    return json.loads(TABLE.read_text()) if TABLE.exists() else {}


def current(p: pd.DataFrame | None = None) -> dict:
    """Every player's shot diet over his last 15 games and what every defense has conceded over its last 15."""
    p = profiles(p if p is not None else load())
    recent = p[p.game_date >= p.game_date.max() - pd.Timedelta(days=45)]
    last = recent.groupby("player_id").tail(WINDOW)
    s, n = last.groupby("name_key")[TYPES + ["min"]].sum(), last.groupby("name_key").size()
    P = pd.DataFrame({"r_" + t: s[t] / s["min"] for t in TYPES})
    P["min_l"] = s["min"] / n
    P = P[n >= MIN_GAMES]
    q = recent.dropna(subset=["r_fga"])
    cols = {}
    for t in TYPES:
        cols["a_" + t], cols["e_" + t] = q[t], q["r_" + t] * q["min"]
    x = pd.concat([q[["opp", "game_id", "game_date"]], pd.DataFrame(cols, index=q.index)], axis=1)
    tg = x.groupby(["opp", "game_id", "game_date"], as_index=False).sum().sort_values("game_date").groupby("opp").tail(WINDOW)
    sums = tg.groupby("opp").sum(numeric_only=True)
    D = pd.DataFrame({"c_" + t: sums["a_" + t] / sums["e_" + t] for t in TYPES})
    D["drop"] = (D.c_pu3 + D.c_pu2) / 2 - (D.c_rim + D.c_cs3) / 2
    return dict(players=P, defenses=D, table=load_table())


def _third(D: pd.DataFrame, col: str, team: str) -> str:
    v = D.loc[team, col]
    return "top" if v > D[col].quantile(2 / 3) else ("bottom" if v < D[col].quantile(1 / 3) else "middle")


def leg_lines(cur: dict, name_key: str, opp: str, series: str) -> tuple[list[str], list[bool]]:
    """(lines, votes) for one leg. A vote is True when the scenario supports the UNDER, False for the OVER; it is
    cast only when the box-score effect is distinguishable from zero (|z| >= 2) with the same sign in both seasons."""
    P, D, T = cur["players"], cur["defenses"], cur.get("table") or {}
    if name_key not in P.index or opp not in D.index or series not in SERIES_STAT or not T:
        return [], []
    row = add_exposure(pd.concat([P.loc[[name_key]].reset_index(drop=True), D.loc[[opp]].reset_index(drop=True)], axis=1))
    types = {k: bool(v.iloc[0]) for k, v in player_types(row).items()}
    r, lines, votes = row.iloc[0], [], []
    stat, xcol = SERIES_STAT[series]
    look = "KXNBAPTS" if series == "KXNBAPRA" else series
    diet = f"{r.sh_rim:.0%} at the rim, {r.sh_mid:.0%} mid-range and floaters, {r.sh_pu3:.0%} pull-up threes, {r.sh_cs3:.0%} catch-and-shoot threes"
    desc = {"pull-up three shooter": f"{r.pu3_36:.1f} pull-up threes per 36 min", "spot-up shooter": f"{r.cs3_36:.1f} catch-and-shoot threes per 36",
            "driver": f"{r.drv_36:.0f} drives per 36", "rim finisher": f"{r.sh_rim:.0%} of his shots at the rim",
            "post scorer": f"{r.post_36:.1f} post touches per 36", "playmaker": f"{r.past_36:.0f} potential assists per 36"}
    hit, found = False, []
    for sid, s in T.get("scenarios", {}).items():
        if not types.get(s["player"]) or look not in s["series"]:
            continue
        third = _third(D, s["col"], opp)
        rank = int(D[s["col"]].rank(ascending=False)[opp])
        kind = s["defense"].split(" (")[0]
        use = stat if stat in s.get("stats", {}) else s["stat"]            # threes for a threes prop, points for a points prop
        st_ = s.get("stats", {}).get(use, dict(diff=s["real_diff"], z=s["real_z"], same_sign=s["same_sign"]))
        d_, z, word = st_["diff"], st_["z"], OTHER[use]
        if third == "middle":
            found.append((0.0, f"scenario: {s['player']} ({desc[s['player']]}) vs {opp}, ranked {rank} of 30 as a {kind}: middle of the pack, no lean", None))
            continue
        fav = (d_ > 0) == (third == "top")                     # does this pairing favour the player?
        proven = abs(z) >= 3 and st_["same_sign"]              # 7 scenarios are tested, so the bar is 3 standard errors, same sign both seasons
        if proven:
            v = "ADVANTAGE" if fav else "DISADVANTAGE"
        elif abs(z) >= 2:
            v = f"leans {'advantage' if fav else 'disadvantage'}, not conclusive"
        else:
            v = "no measurable effect"
        priced = "Kalshi's price already reflects it" if abs(s["mk_z"]) < 2 else f"overs ran {s['mk_diff']:+.1%} against the price in the top third (unconfirmed)"
        found.append((abs(z), f"scenario: {s['player']} ({desc[s['player']]}) vs {opp}, ranked {rank} of 30 ({third} third) as a {kind}. "
                              f"Across {s['games']:,} games such players' {word} ran {d_:+.1%} against top-third defenses compared with bottom-third ones "
                              f"(z {z:+.1f}) -> tonight: {v}; {priced}",
                      (not fav) if proven else None))
    for _, text, v_ in sorted(found, key=lambda t: -t[0])[:2]:
        lines.append(text)
        hit = True
        if v_ is not None:
            votes.append(v_)
    e = T.get("exposure", {}).get(xcol)
    if e and pd.notna(r[xcol]):
        x = float(r[xcol])
        third = "favourable" if x > e["q67"] else ("unfavourable" if x < e["q33"] else "neutral")
        solid = e["z_last"] >= 3 and e["z_this"] >= 3
        base = (f"shot diet ({diet}) against what {opp} concedes: {x:+.1%} expected looks" if xcol in ("x_pts", "x_3pt")
                else f"{opp} concedes {x:+.1%} {'potential assists' if xcol == 'x_ast' else 'rebound chances'} vs what its opponents normally get")
        tail = (f"; in both seasons the favourable third out-produced the unfavourable third ({e['diff_last']:+.1%}, {e['diff_this']:+.1%}), and Kalshi's price already reflected it"
                if solid else "; this measure did not hold up in both seasons")
        if not hit or third != "neutral":
            lines.append(f"{base} ({third} third){tail}")
        if solid and third != "neutral":
            votes.append(third == "unfavourable")
    return lines, votes


def run(args=None) -> None:
    res = study()
    save_table(res)
    try:
        from . import nfl_scenarios
        nfl_scenarios.study()
    except Exception as e:
        print(f"\n(NFL coverage study unavailable: {e})")
