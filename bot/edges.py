"""Systematic hunt for leg types that hit more often than they cost — with the statistics that
stop us fooling ourselves.

Scanning hundreds of (market type x side x price band x snapshot) "angles" guarantees some look
great by luck. So:
  1. split nights in two: DISCOVER on the first half, CONFIRM on the second (never touched before);
  2. test each angle with errors clustered by game (legs in one game are not independent);
  3. control the false-discovery rate across all angles (Benjamini-Hochberg, statsmodels);
  4. an angle only counts if it is significant in discovery AND still positive after fees in
     confirmation.
Confirmed angles are saved to data/edges.json and used by the card builders.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config
from .favorites import legs_table
from .kalshi import fee

EDGES_PATH = config.DATA_DIR / "edges.json"
BANDS = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.995]


def _stat(g: pd.DataFrame) -> dict:
    net = g.win - g.cost - g.cost.map(fee)
    cl = net.groupby(g.game).sum()
    n, G = len(g), len(cl)
    se = np.sqrt((cl ** 2).sum() - cl.sum() ** 2 / G) / n * np.sqrt(G / max(G - 1, 1)) if G > 1 else np.nan
    mean = net.mean()
    t = mean / se if se and se > 0 else 0.0
    return dict(n=n, games=G, price=g.cost.mean(), hit=g.win.mean(), net=mean, roi=net.sum() / g.cost.sum(),
                se=se, t=t, p=float(1 - st.t.cdf(t, max(G - 1, 1))))


def scan(hist: pd.DataFrame, snapshots=("pre", "live1", "live2"), min_n: int = 150, q: float = 0.10,
         max_spread: float = 0.15) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    rows = []
    for snap in snapshots:
        if f"{snap}_ask" not in hist.columns or hist[f"{snap}_ask"].notna().sum() == 0:
            continue
        legs = legs_table(hist, snap, max_spread=max_spread)
        nights = sorted(legs.date.unique())
        cut = nights[len(nights) // 2]
        legs["band"] = pd.cut(legs.cost, BANDS, right=False)
        for (series, side, band), g in legs.groupby(["series", "side", "band"], observed=True):
            a, b = g[g.date < cut], g[g.date >= cut]
            if len(a) < min_n or len(b) < min_n:
                continue
            d, c = _stat(a), _stat(b)
            rows.append(dict(snapshot=snap, series=series, side=side, band=str(band),
                             d_n=d["n"], d_price=d["price"], d_hit=d["hit"], d_roi=d["roi"], d_t=d["t"], d_p=d["p"],
                             c_n=c["n"], c_hit=c["hit"], c_price=c["price"], c_roi=c["roi"], c_t=c["t"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr_pass"] = multipletests(df.d_p, alpha=q, method="fdr_bh")[0]
    df["confirmed"] = df.fdr_pass & (df.c_roi > 0) & (df.c_t > 1.0)
    return df.sort_values("d_t", ascending=False)


def run(hist: pd.DataFrame):
    pd.set_option("display.width", 220)
    df = scan(hist)
    if df.empty:
        print("not enough history yet")
        return df
    k = len(df)
    print(f"\n{k} angles tested (market type x YES/NO x price band x snapshot), each needing 150+ legs in both halves.")
    print(f"By luck alone about {0.05 * k:.0f} would show t > 1.65 in discovery. Found: {(df.d_t > 1.65).sum()}.")
    print(f"Pass false-discovery control (q=10%): {int(df.fdr_pass.sum())}.  Also hold up in the untouched half: {int(df.confirmed.sum())}.")
    cols = ["snapshot", "series", "side", "band", "d_n", "d_price", "d_hit", "d_roi", "d_t", "c_n", "c_hit", "c_roi", "c_t",
            "fdr_pass", "confirmed"]
    print("\nBest-looking angles in the discovery half, and what they did in the confirmation half:")
    print(df[cols].head(12).round(3).to_string(index=False))
    print("\nDoes discovery predict confirmation at all? correlation of ROI across halves: "
          f"{df.d_roi.corr(df.c_roi):+.2f}  (near 0 = pure noise; positive = angles differ consistently)")
    bad = df[(df.d_t < -2) & (df.c_t < -2)].sort_values("c_roi")
    if len(bad):
        print(f"\nConsistently LOSING angles (t < -2 in both halves) - the card builders steer away from these:")
        print(bad[["snapshot", "series", "side", "band", "d_roi", "c_roi"]].head(10).round(3).to_string(index=False))
    conf = df[df.confirmed]
    EDGES_PATH.write_text(json.dumps(conf[["snapshot", "series", "side", "band", "c_roi"]].to_dict("records"), indent=1))
    return df
