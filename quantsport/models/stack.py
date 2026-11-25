"""Does the rating model add anything to Kalshi's price? And the probability the fund actually uses.

For every settled Kalshi game market with a pre-game price (winner, spread, total, team total; NBA, NFL, college
football) the rating model's probability is computed from ratings as they stood before that game. Then:

    result ~ a + b_price x logit(Kalshi price) + b_model x logit(model probability)

is fitted on the EARLIER part of the data and scored on the later part. b_model near zero means the model knows
nothing the price does not. The fitted blend, with the uncertainty of its coefficients, is what the decision
chain uses as "our probability" and "how sure are we": never the raw model.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from bot import config
from bot.favorites import slate_date

from .. import gamedata as G
from . import ratings as R

MODELS = config.DATA_DIR / "quantsport_models.json"
TABLE = config.DATA_DIR / "quantsport_game_markets.parquet"
TRAIN_UNTIL = {"NBA": "2025-10-01", "NFL": "2025-08-01", "NCAAF": "2026-08-01"}     # ratings parameters are tuned before these dates


def _logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def fit_ratings() -> dict:
    """Tune and run the rating model for each sport. Returns {sport: (per-game predictions, final Ratings, params)}."""
    out = {}
    for sport in ("NBA", "NFL", "NCAAF"):
        g = G.results(sport)
        b = R.tune(sport, g, TRAIN_UNTIL[sport])
        par = {k: b[k] for k in ("k", "hfa", "carry")}
        d, rt = R.walk(sport, g, **par)
        out[sport] = (d, rt, par)
    return out


def _split_code(sport: str, code: str, date, games: pd.DataFrame) -> tuple[str, str] | None:
    """(away, home) in the results table's own team names for a Kalshi game code such as 26NOV23MILTOR."""
    teams = code[7:]
    day = games[(games.date >= date - pd.Timedelta(days=1)) & (games.date <= date + pd.Timedelta(days=1))]
    inv = {v: k for k, v in G.NFL_ALIAS.items()}
    for g in day.itertuples():
        for a in {g.away, inv.get(g.away, g.away)}:
            for h in {g.home, inv.get(g.home, g.home)}:
                if teams == a + h:
                    return g.away, g.home
    return None


def market_table(fits: dict | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Every settled Kalshi game market joined to the model's pre-game probability for it."""
    if TABLE.exists() and not rebuild:
        return pd.read_parquet(TABLE)
    fits = fits or fit_ratings()
    pre = G.build_pre()
    pre["date"] = pd.to_datetime(pre.game.map(slate_date))
    seg = pre.ticker.str.split("-").str[-1]
    pre["team_code"] = seg.str.extract(r"^([A-Z]+)")[0]
    rows = []
    for sport, (pred, _, _) in fits.items():
        m = pre[pre.league == sport]
        if m.empty:
            continue
        pred = pred.copy()
        if sport == "NCAAF":       # names: winner-market titles ("Oklahoma St. wins") matched to schedule names on that date
            names = {}
            win = m[m.kind == "winner"]
            for (game, date), e in win.groupby(["game", "date"]):
                day = pred[(pred.date >= date - pd.Timedelta(days=1)) & (pred.date <= date + pd.Timedelta(days=1))]
                opts = list(day.home) + list(day.away)
                got = {r.team_code: G.match_name(r.title.replace(" wins", ""), opts) for r in e.itertuples()}
                if len(got) == 2 and all(s >= 0.8 for _, s in got.values()):
                    a, b = [v[0] for v in got.values()]
                    hit = day[((day.home == a) & (day.away == b)) | ((day.home == b) & (day.away == a))]
                    if len(hit) == 1:
                        names[game] = ({c: v[0] for c, v in got.items()}, hit.iloc[0])
            for r in m.itertuples():
                if r.game in names and r.team_code in names[r.game][0]:
                    g = names[r.game][1]
                    rows.append((r, g, names[r.game][0][r.team_code] == g.home))
        else:
            cache = {}
            for r in m.itertuples():
                if r.game not in cache:
                    ah = _split_code(sport, r.game, r.date, pred)
                    cache[r.game] = None
                    if ah:
                        hit = pred[(pred.away == ah[0]) & (pred.home == ah[1]) & (pred.date >= r.date - pd.Timedelta(days=1)) & (pred.date <= r.date + pd.Timedelta(days=1))]
                        if len(hit):
                            cache[r.game] = hit.iloc[0]
                g = cache[r.game]
                if g is None:
                    continue
                inv = {v: k for k, v in G.NFL_ALIAS.items()}
                is_home = r.team_code in (g.home, inv.get(g.home, g.home)) if isinstance(r.team_code, str) else None
                rows.append((r, g, is_home))
    out = []
    for r, g, is_home in rows:
        P = R.Pred(g.mu, g.sigma, g.total, g.sigma_t, int(g.n_home), int(g.n_away))
        line = float(r.floor_strike) if pd.notna(r.floor_strike) else np.nan
        if r.kind == "winner":
            pm = P.p_win(bool(is_home))
        elif r.kind == "spread" and line == line:
            pm = P.p_margin_over(line, bool(is_home))
        elif r.kind == "total" and line == line:
            pm = P.p_total_over(line)
        elif r.kind == "team_total" and line == line:
            pm = P.p_team_over(line, bool(is_home))
        else:
            continue
        out.append(dict(ticker=r.ticker, sport=r.league, kind=r.kind, game=r.game, date=r.date, title=r.title, line=line, p_mkt=r.pre,
                        p_model=pm, y=r.y, n_games=min(int(g.n_home), int(g.n_away))))
    d = pd.DataFrame(out)
    d.to_parquet(TABLE, index=False)
    return d


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(d: pd.DataFrame | None = None, save: bool = True, verbose: bool = True) -> dict:
    """Per sport and market type: model vs price, the walk-forward blend, and the coefficients kept for live use."""
    import statsmodels.api as sm
    d = d if d is not None else market_table()
    d = d[(d.p_mkt > 0.05) & (d.p_mkt < 0.95) & (d.n_games >= 5)].sort_values("date")
    res, rows = {}, []
    for (sport, kind), g in d.groupby(["sport", "kind"]):
        if len(g) < 300 or g.game.nunique() < 60:
            continue
        cut = g.date.quantile(0.6)
        tr, te = g[g.date <= cut], g[g.date > cut]
        if len(te) < 100:
            continue
        X = lambda x: sm.add_constant(pd.DataFrame({"price": _logit(x.p_mkt), "model": _logit(x.p_model)}), has_constant="add")
        f = sm.Logit(tr.y.values, X(tr)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(tr.game)[0]})
        only = sm.Logit(tr.y.values, X(tr)[["const", "price"]]).fit(disp=0)
        full = sm.Logit(g.y.values, X(g)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.game)[0]})
        row = dict(sport=sport, market=kind, markets=len(g), games=g.game.nunique(), model_logloss=_ll(g.p_model.values, g.y.values), price_logloss=_ll(g.p_mkt.values, g.y.values),
                   b_price=float(f.params["price"]), b_model=float(f.params["model"]), z_model=float(f.tvalues["model"]),
                   test_markets=len(te), test_price=_ll(te.p_mkt.values, te.y.values),
                   test_recal=_ll(np.asarray(only.predict(X(te)[["const", "price"]])), te.y.values),
                   test_blend=_ll(np.asarray(f.predict(X(te))), te.y.values))
        row["recal_helps"] = bool(row["test_recal"] < row["test_price"] - 1e-4)
        row["blend_helps"] = bool(row["test_blend"] < min(row["test_price"], row["test_recal"]) - 1e-4 and row["z_model"] > 2)
        rows.append(row)
        res[f"{sport}:{kind}"] = dict(params=[float(v) for v in full.params], cov=np.asarray(full.cov_params()).tolist(), n=len(g),
                                      z_model=float(full.tvalues["model"]), helps=row["blend_helps"],
                                      test_gain=row["test_price"] - row["test_blend"], model_logloss=row["model_logloss"],
                                      price_logloss=row["price_logloss"], markets=len(g))
    tab = pd.DataFrame(rows)
    if save:
        cur = json.loads(MODELS.read_text()) if MODELS.exists() else {}
        cur["stack"], cur["made"] = res, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        MODELS.write_text(json.dumps(cur, indent=1))
    if verbose and len(tab):
        pd.set_option("display.width", 230)
        print(f"\nRATING MODEL vs KALSHI'S PRE-GAME PRICE on {len(d):,} settled game markets (lower log-loss = better).")
        print("b_model = weight the fitted blend puts on the model after the price (z over 2 = it knows something the price does not).")
        print("test_* = fitted on the earlier 60% of dates, scored on the later 40%.\n")
        print(tab.round(4).to_string(index=False))
    return dict(table=tab, stack=res)


def estimate(sport: str, kind: str, p_mkt: float, p_model: float | None) -> tuple[float, float, str]:
    """(our probability, its standard error, how it was made).

    The blend is used ONLY where it beat the price on later data with a model weight over 2 standard errors. Everywhere
    else our probability is Kalshi's price: an unproven model moving the estimate would manufacture edges out of noise.
    The model's own number is still reported on the thesis as supporting or contradicting evidence."""
    st = (json.loads(MODELS.read_text()).get("stack", {}) if MODELS.exists() else {}).get(f"{sport}:{kind}")
    if st is None or p_model is None or not np.isfinite(p_model):
        return float(p_mkt), 0.03, "Kalshi's price (no tested model for this market type)"
    if not st["helps"]:
        return float(p_mkt), 0.03, (f"Kalshi's price. The rating model was tested on {st['markets']:,} settled {sport} {kind.replace('_', ' ')} markets and did not "
                                    f"improve on the price (log-loss {st['model_logloss']:.3f} vs {st['price_logloss']:.3f}; weight after the price {st['params'][2]:+.2f}, z {st['z_model']:+.1f})")
    x = np.array([1.0, _logit(p_mkt), _logit(p_model)])
    b, cov = np.array(st["params"]), np.array(st["cov"])
    p = 1 / (1 + np.exp(-float(x @ b)))
    se = float(np.sqrt(max(x @ cov @ x, 0)) * p * (1 - p))
    return float(p), max(se, 0.01), f"blend of Kalshi's price and the rating model (model weight {b[2]:+.2f}, z {st['z_model']:+.1f}; it beat the price on later data)"


def live(sports=("NBA", "NFL", "NCAAF")) -> dict:
    """Ratings as of now for each sport, with the tuned parameters saved by `fit_ratings` (re-tuned if missing)."""
    cur = json.loads(MODELS.read_text()) if MODELS.exists() else {}
    par = cur.get("ratings", {})
    out = {}
    for sport in sports:
        g = G.results(sport)
        if sport not in par:
            b = R.tune(sport, g, TRAIN_UNTIL[sport])
            par[sport] = {k: b[k] for k in ("k", "hfa", "carry")}
        _, rt = R.walk(sport, g, **par[sport])
        out[sport] = rt
    cur["ratings"] = par
    MODELS.write_text(json.dumps(cur, indent=1))
    return out
