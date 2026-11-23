"""NFL scenarios: does a receiver go over against man or against zone?

The bettor's idea: some receivers win against man coverage, others find the holes in zone, so a receiver's
yards should depend on what tonight's defense plays. Tested on every charted dropback of 2024 and 2025.

Data: nflverse (github.com/nflverse/nflverse-data releases; play-by-play CC-BY 4.0, participation charted
by FTN, CC-BY-SA 4.0 "FTN Data via nflverse"). Each dropback carries the defense's man/zone call, the
players on the field and the result. Downloaded at run time into data/ext/nfl, never committed.

LIMIT: the charted coverage for a season is published only after that season's Super Bowl. So this can
be TESTED on 2025 (with 2024 as history) but cannot be applied live to 2026 games: there is no 2026
man/zone data to know how a defense is playing this year.

Walk-forward: a game only sees earlier games (2024 counts as history for early 2025).
  receiver split   yards per dropback he was on the field for, vs man and vs zone, shrunk toward his own
                   overall rate (80 dropbacks of prior) so a few big plays do not define him
  defense          share of its dropbacks in man over its previous 8 games
  exposure         (defense's man share - league share) x (his man rate - his zone rate) / his overall rate:
                   the % more yards per route he should get if his split is real and the defense stays itself
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from . import config
from .data import norm_name

EXT = config.DATA_DIR / "ext" / "nfl"
REL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{name}.parquet"
PRIOR_K, DEF_WINDOW = 80.0, 8


def _get(tag: str, name: str) -> pd.DataFrame:
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / f"{name}.parquet"
    if not f.exists():
        r = requests.get(REL.format(tag=tag, name=name), timeout=600)
        r.raise_for_status()
        f.write_bytes(r.content)
    return pd.read_parquet(f)


def dropbacks(seasons=(2024, 2025)) -> pd.DataFrame:
    """One row per charted dropback: defense, man/zone, who was on the field, who caught what."""
    parts = []
    for y in seasons:
        b = _get("pbp", f"play_by_play_{y}")
        b = b[(b.qb_dropback == 1) & (b.season_type == "REG")][["game_id", "play_id", "game_date", "week", "posteam", "defteam",
                                                                "receiver_player_id", "receiving_yards", "complete_pass", "pass_attempt"]]
        pp = _get("pbp_participation", f"pbp_participation_{y}")[["nflverse_game_id", "play_id", "defense_man_zone_type", "offense_players"]]
        pp = pp.rename(columns={"nflverse_game_id": "game_id", "defense_man_zone_type": "mz"})
        d = b.merge(pp, on=["game_id", "play_id"], how="inner")
        d = d[d.mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE", "MAN", "ZONE"])]
        d["man"] = d.mz.str.startswith("MAN").astype(float)
        parts.append(d.assign(season=y))
    d = pd.concat(parts, ignore_index=True)
    d["game_date"] = pd.to_datetime(d.game_date)
    d["receiving_yards"] = d.receiving_yards.fillna(0.0)
    return d


def receiver_games(d: pd.DataFrame) -> pd.DataFrame:
    """Per receiver and game: dropbacks on the field and yards, split by coverage."""
    on = d[["game_id", "play_id", "game_date", "season", "defteam", "man", "offense_players"]].copy()
    on["player_id"] = on.offense_players.fillna("").str.split(";")
    on = on.explode("player_id")
    on = on[on.player_id.str.len() > 0]
    tg = d[["game_id", "play_id", "receiver_player_id", "receiving_yards", "complete_pass"]].rename(columns={"receiver_player_id": "player_id"})
    on = on.merge(tg, on=["game_id", "play_id", "player_id"], how="left")
    on["target"] = on.receiving_yards.notna().astype(float)
    on["yards"], on["rec"] = on.receiving_yards.fillna(0.0), on.complete_pass.fillna(0.0)
    for c in ("yards", "target", "rec"):
        on[c + "_man"], on[c + "_zone"] = on[c] * on.man, on[c] * (1 - on.man)
    on["db"], on["db_man"], on["db_zone"] = 1.0, on.man, 1 - on.man
    cols = ["db", "db_man", "db_zone", "yards", "yards_man", "yards_zone", "target", "target_man", "target_zone", "rec"]
    g = on.groupby(["player_id", "game_id", "game_date", "season", "defteam"], as_index=False)[cols].sum()
    return g.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def build() -> pd.DataFrame:
    d = dropbacks()
    g = receiver_games(d)
    # what the receiver had done BEFORE each game (all earlier games, both seasons)
    grp = g.groupby("player_id")
    for c in ("db", "db_man", "db_zone", "yards", "yards_man", "yards_zone", "target"):
        g["p_" + c] = grp[c].cumsum() - g[c]
    g["games_prior"] = grp.cumcount()
    g["ypd"] = g.p_yards / g.p_db.where(g.p_db > 0)
    g["ypd_man"] = (g.p_yards_man + PRIOR_K * g.ypd) / (g.p_db_man + PRIOR_K)
    g["ypd_zone"] = (g.p_yards_zone + PRIOR_K * g.ypd) / (g.p_db_zone + PRIOR_K)
    g["split"] = (g.ypd_man - g.ypd_zone) / g.ypd.where(g.ypd > 0)            # + = better against man
    # the defense's man share over its previous games
    t = d.groupby(["defteam", "game_id", "game_date"], as_index=False).agg(man=("man", "sum"), n=("man", "size")).sort_values(["defteam", "game_date"])
    tg = t.groupby("defteam")
    t["def_man"] = tg.man.transform(lambda s: s.shift(1).rolling(DEF_WINDOW, min_periods=3).sum()) / \
        tg.n.transform(lambda s: s.shift(1).rolling(DEF_WINDOW, min_periods=3).sum())
    t["game_man"] = t.man / t.n
    lg = t.groupby("game_date").agg(m=("man", "sum"), n=("n", "sum")).sort_index()
    lg["lg_man"] = (lg.m.cumsum() - lg.m) / (lg.n.cumsum() - lg.n).where(lambda v: v > 0)
    g = g.merge(t[["defteam", "game_id", "def_man", "game_man"]], on=["defteam", "game_id"], how="left").merge(lg[["lg_man"]], left_on="game_date", right_index=True, how="left")
    g["exposure"] = (g.def_man - g.lg_man) * g.split
    g["rel"] = (g.yards / g.db.where(g.db > 0)) / g.ypd.where(g.ypd > 0) - 1      # this game's yards per route vs his own norm
    pl = _get("players", "players")
    idc = "gsis_id" if "gsis_id" in pl.columns else "player_id"
    pl = pl[[idc, "display_name", "position"]].rename(columns={idc: "player_id"}).drop_duplicates("player_id")
    g = g.merge(pl, on="player_id", how="left")
    g["name_key"] = g.display_name.fillna("").map(norm_name)
    return g[g.position.isin(["WR", "TE", "RB"])]


def kalshi_markets() -> pd.DataFrame:
    """Settled Kalshi receiving-yards and receptions markets with a pre-game price."""
    from .favorites import slate_date
    from .trades import PRIOR_RESULTS, RESULTS
    parts = []
    for pre, res in (("prop_pre_prior.parquet", PRIOR_RESULTS), ("prop_pre.parquet", RESULTS)):
        p = pd.read_parquet(config.DATA_DIR / pre)
        p = p[p.series.isin(["KXNFLRECYDS", "KXNFLREC"])]
        w = p.pivot_table(index=["ticker", "series", "game", "result"], columns="taker_side", values="cost").reset_index()
        for c in ("yes", "no"):
            if c not in w:
                w[c] = np.nan
        w["p_mkt"] = np.where(w.yes.notna() & w.no.notna(), (w.yes + 1 - w.no) / 2, w.yes.fillna(1 - w.no))
        r = pd.read_parquet(res)[["ticker", "title", "floor_strike"]]
        parts.append(w.merge(r, on="ticker"))
    m = pd.concat(parts, ignore_index=True)
    m["name_key"] = m.title.str.replace(r"(:| records ).*$", "", regex=True).map(norm_name)     # both seasons' title styles
    m["game_date"] = pd.to_datetime(m.game.map(slate_date))
    m["y"] = (m.result == "yes").astype(float)
    return m.dropna(subset=["game_date", "p_mkt"])


def _mean_se(x, groups):
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return float("nan"), float("nan"), len(x)
    s = (x - x.mean()).groupby(groups.loc[x.index]).sum()
    return float(x.mean()), float(np.sqrt((s ** 2).sum()) / len(x)), len(x)


def _cmp(x, a, b, groups):
    ma, sa, na = _mean_se(x[a], groups)
    mb, sb, nb = _mean_se(x[b], groups)
    se = np.sqrt(sa ** 2 + sb ** 2)
    return ma, mb, ma - mb, (ma - mb) / se if se > 0 else float("nan"), na, nb


def study(verbose: bool = True) -> dict:
    g = build()
    q = g[(g.season == 2025) & (g.p_db >= 150) & (g.p_db_man >= 40) & (g.db >= 15) & g.def_man.notna() & (g.ypd > 0.4)].copy()
    out = {}
    # how stable is a receiver's man/zone split, and a defense's man share?
    h = g[(g.season == 2025)].groupby("player_id").agg(db_man=("db_man", "sum"), db_zone=("db_zone", "sum"), ym=("yards_man", "sum"), yz=("yards_zone", "sum"),
                                                         wk=("game_date", "rank"))
    first = g[(g.season == 2025)].assign(half=lambda x: np.where(x.game_date < x.game_date.quantile(0.5), "a", "b"))
    s = first.groupby(["player_id", "half"]).agg(dm=("db_man", "sum"), dz=("db_zone", "sum"), ym=("yards_man", "sum"), yz=("yards_zone", "sum"))
    s = s[(s.dm >= 40) & (s.dz >= 80)]
    s["split"] = s.ym / s.dm - s.yz / s.dz
    w = s.split.unstack().dropna()
    out["split_stability"] = (float(w.corr().iloc[0, 1]), len(w))
    out["def_stability"] = float(q[["def_man", "game_man"]].corr().iloc[0, 1])
    hi_d, lo_d = q.def_man > q.def_man.quantile(2 / 3), q.def_man < q.def_man.quantile(1 / 3)
    manb, zoneb = q.split > q.split.quantile(2 / 3), q.split < q.split.quantile(1 / 3)
    fav, unf = q.exposure > q.exposure.quantile(2 / 3), q.exposure < q.exposure.quantile(1 / 3)
    out["real"] = {
        "man-beater: man-heavy defense vs zone-heavy defense": _cmp(q.rel, manb & hi_d, manb & lo_d, q.game_id),
        "zone-beater: zone-heavy defense vs man-heavy defense": _cmp(q.rel, zoneb & lo_d, zoneb & hi_d, q.game_id),
        "all receivers: favourable third of coverage exposure vs unfavourable third": _cmp(q.rel, fav, unf, q.game_id),
        "any receiver, whatever his split: zone-heavy defense vs man-heavy defense": _cmp(q.rel, lo_d, hi_d, q.game_id)}
    m = kalshi_markets().merge(q[["name_key", "game_date", "exposure", "split", "def_man", "game_id"]].drop_duplicates(["name_key", "game_date"]),
                               on=["name_key", "game_date"], how="inner")
    m = m[(m.p_mkt > 0.08) & (m.p_mkt < 0.92)]
    out["markets"] = len(m)
    gap = m.y - m.p_mkt
    mf, mu = m.exposure > m.exposure.quantile(2 / 3), m.exposure < m.exposure.quantile(1 / 3)
    out["priced"] = _cmp(gap, mf, mu, m.game_id)
    coef = z = float("nan")
    if len(m) > 500:
        import statsmodels.api as sm
        X = pd.DataFrame({"price": np.log(m.p_mkt / (1 - m.p_mkt)), "x": (m.exposure - m.exposure.mean()) / m.exposure.std()})
        f = sm.Logit(m.y.values, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(m.game_id)[0]})
        coef, z = float(f.params["x"]), float(f.tvalues["x"])
    out["beyond_price"] = (coef, z)
    out["n"] = len(q)
    if verbose:
        print(f"\nNFL SCENARIOS: receivers against man and zone coverage. 2025 regular season, {len(q):,} receiver-games "
              f"(150+ charted routes of history, 15+ routes in the game), {len(m):,} settled Kalshi receiving markets.")
        r, n = out["split_stability"]
        print(f"  Is a receiver's man/zone split a stable trait? First half of 2025 vs second half, {n} receivers: correlation {r:+.2f} "
              f"(1 = fully repeatable, 0 = noise).")
        print(f"  Is a defense's coverage predictable? Its previous-8-game man share vs what it played that game: correlation {out['def_stability']:+.2f}.")
        print("  REAL? yards per route in the game against the receiver's own norm:")
        for k, (a, b, dff, zz, na, nb) in out["real"].items():
            print(f"     {k:76s} {a:+.1%} vs {b:+.1%}   diff {dff:+.1%} (z {zz:+.1f}; {na}/{nb} games)")
        a, b, dff, zz, na, nb = out["priced"]
        print(f"  PRICED? Kalshi overs against their price, favourable third vs unfavourable third: {a:+.1%} vs {b:+.1%}, diff {dff:+.1%} (z {zz:+.1f}; {na}/{nb} markets)")
        print(f"          one standard deviation of exposure adds {coef:+.3f} to the odds of the over after the price (z {z:+.1f}).")
        print("  Coverage charting for 2026 is not published until after the season, so this cannot be applied to this year's games.")
    return out
