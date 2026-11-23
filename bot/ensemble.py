"""Do our models know anything Kalshi's price doesn't?  Tested on Kalshi's own settled props.

For every settled player-prop market with a pre-game quote we rebuild, as of that morning, the
three model probabilities (p_nb, p_hit, p_gbm), then
  * score each against the real outcome next to the Kalshi mid (Brier / log-loss),
  * fit a walk-forward stacker  logit(p) ~ logit(mid) + logit(models)  using only earlier nights,
  * simulate betting the stacker against the real bid/ask after fees.
If the stacker can't beat the mid out-of-sample, no combo built on these models has an edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .favorites import slate_date
from .features import FEATS, leg_features, load_gbm
from .kalshi import fee
from .markets import PROP_SERIES
from .model import Model, load_priors

PROPS_PATH = config.DATA_DIR / "history_props_scored.parquet"
WEIGHTS_PATH = config.DATA_DIR / "stack_weights.json"
MODELS = ["p_nb", "p_hit", "p_gbm"]


def _logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return np.log(p / (1 - p))


def score_history(hist: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    h = hist[hist.series.isin(PROP_SERIES) & hist.result.isin(["yes", "no"]) & hist.pre_ask.notna()
             & hist.pre_bid.notna() & (hist.strike_type == "greater")].copy()
    h["date"] = h.game.map(slate_date)
    pri, gbm, rows = load_priors(logs), load_gbm(), []
    for d, day in h.groupby("date"):
        m = Model(logs, as_of=d, priors=pri)
        for r in day.itertuples():
            seg = r.ticker.split("-")[-2]
            team, away, home = seg[:3], r.game[-6:-3], r.game[-3:]
            if team not in (away, home):
                continue
            f = leg_features(m, r.title.split(":")[0].strip(), team, home if team == away else away,
                             team == home, PROP_SERIES[r.series], r.floor_strike)
            if f is None:
                continue
            f.update(ticker=r.ticker, series=r.series, game=r.game, date=d, bid=r.pre_bid, ask=r.pre_ask,
                     y=float(r.result == "yes"), volume=r.volume)
            rows.append(f)
    df = pd.DataFrame(rows)
    df["mid"] = np.where((df.bid > 0) & (df.ask - df.bid <= 0.04), (df.bid + df.ask) / 2,
                         df.bid + 0.65 * (df.ask - df.bid))
    df["p_gbm"] = gbm.predict_proba(df[FEATS].values)[:, 1] if gbm is not None else np.nan
    df.to_parquet(PROPS_PATH, index=False)
    return df


def stack_walk_forward(df: pd.DataFrame, warmup: int = 8, cols=None) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    cols = cols or ["mid"] + [c for c in MODELS if df[c].notna().all()]
    df = df.sort_values("date").copy()
    X = np.column_stack([_logit(df[c]) for c in cols])
    df["p_stack"] = np.nan
    dates = sorted(df.date.unique())
    coefs = None
    for d in dates[warmup:]:
        tr, te = (df.date < d).values, (df.date == d).values
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], df.y.values[tr])
        df.loc[te, "p_stack"] = lr.predict_proba(X[te])[:, 1]
        coefs = dict(zip(cols, lr.coef_[0].round(3)))
    df.attrs["coefs"] = coefs
    return df


def report(df: pd.DataFrame) -> None:
    ev = df.dropna(subset=["p_stack"])
    y = ev.y
    print(f"\nPLAYER PROPS: {len(df):,} settled legs with a pre-game quote and a projectable player; "
          f"{len(ev):,} scored out-of-sample over {ev.date.nunique()} nights")
    print(f"  {'predictor':10s} {'Brier':>8s} {'log-loss':>9s}")
    for c in ["mid", *MODELS, "p_stack"]:
        if ev[c].notna().all():
            p = np.clip(ev[c], 0.005, 0.995)
            print(f"  {c:10s} {((p - y) ** 2).mean():8.5f} {-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean():9.5f}")
    print(f"  stacker weights on logits (last fit): {df.attrs.get('coefs')}")
    print("\n  Betting the stacker vs the real quote, after fees (out-of-sample):")
    print(f"  {'min edge':>8s} {'bets':>7s} {'win%':>6s} {'avg price':>9s} {'ROI':>8s} {'+/-':>6s}")
    for thr in (0.0, 0.02, 0.04, 0.06, 0.10):
        yes = ev[ev.p_stack - ev.ask - ev.ask.map(fee) > thr]
        no = ev[(ev.bid > 0) & ((1 - ev.p_stack) - (1 - ev.bid) - (1 - ev.bid).map(fee) > thr)]
        cost = pd.concat([yes.ask, 1 - no.bid])
        win = pd.concat([yes.y, 1 - no.y])
        game = pd.concat([yes.game, no.game])
        if len(cost) < 20:
            continue
        net = win - cost - cost.map(fee)
        g = net.groupby(game.values).sum()
        se = np.sqrt((g ** 2).sum()) / cost.sum()
        print(f"  {thr:8.2f} {len(cost):7d} {win.mean():6.1%} {cost.mean():9.3f} {net.sum() / cost.sum():+8.1%} {1.96 * se:6.1%}")


def fit_live_weights(df: pd.DataFrame) -> dict:
    """Final weights for the live scanner: logit(p) = b0 + w_mid*logit(mid) + w_model*logit(p_nb),
    fitted on ALL settled props. The live scan then trusts the model exactly as much as history says."""
    import json
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([_logit(df.mid), _logit(df.p_nb)])
    lr = LogisticRegression(C=1.0, max_iter=500).fit(X, df.y.values)
    w = dict(b0=float(lr.intercept_[0]), w_mid=float(lr.coef_[0][0]), w_model=float(lr.coef_[0][1]),
             n=int(len(df)), nights=int(df.date.nunique()))
    WEIGHTS_PATH.write_text(json.dumps(w, indent=1))
    return w
