"""Our probability for every scanned side, with a standard error and a plain statement of where it came from.

Three sources, in order of trust:
  1. a blend of price and model that beat the price on later data          (none has, as of the last test)
  2. a graded leg category that has paid in both seasons (bot/legs.py)      -> the price plus that category's measured gap
  3. Kalshi's own price                                                      -> no edge by construction

The raw rating-model number is always attached as evidence, never used as the estimate unless it earned (1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import legs as botlegs

from . import gamedata as G
from . import mathx as M
from .models import ratings as R
from .models import stack


def _game_teams(d: pd.DataFrame, rt: dict) -> dict:
    """{game_key: (away, home, {kalshi team code: ratings name})} for every scanned game we can identify."""
    out = {}
    inv = {v: k for k, v in G.NFL_ALIAS.items()}
    win = d[(d.kind == "winner") & (d.side == "yes")]
    for (sport, game), e in win.groupby(["sport", "game"]):
        teams, known = game[7:], rt[sport].r
        codes = list(e.team_code.unique())
        if len(codes) != 2:
            continue
        away_c = codes[0] if teams.startswith(codes[0]) and not (teams.startswith(codes[1]) and len(codes[1]) > len(codes[0])) else codes[1]
        home_c = codes[1] if away_c == codes[0] else codes[0]
        name = {}
        for c in (away_c, home_c):
            if sport == "NCAAF":
                title = e[e.team_code == c].title.iloc[0].replace(" wins", "")
                nm, score = G.match_name(title, list(known))
                if score >= 0.82:
                    name[c] = nm
            else:
                nm = G.NFL_ALIAS.get(c, c) if sport == "NFL" else c
                if nm in known:
                    name[c] = nm
        if len(name) == 2:
            out[f"{sport}:{game}"] = (name[away_c], name[home_c], {away_c: name[away_c], home_c: name[home_c]})
    return out


def model_probs(d: pd.DataFrame, rt: dict | None = None) -> pd.DataFrame:
    """Adds p_model: the rating model's chance that each game-market side pays (NaN for props and unidentified games)."""
    rt = rt or stack.live(tuple(d.sport.unique()))
    teams = _game_teams(d, rt)
    neutral = {}
    try:
        up = G.upcoming("NCAAF")
        neutral = {(r.away, r.home): bool(r.neutral) for r in up.itertuples()}
    except Exception:
        pass
    preds, pm = {}, np.full(len(d), np.nan)
    for i, r in enumerate(d.itertuples()):
        if r.kind not in ("winner", "spread", "total", "team_total") or r.game_key not in teams:
            continue
        away, home, codes = teams[r.game_key]
        if r.game_key not in preds:
            preds[r.game_key] = rt[r.sport].predict(home, away, neutral.get((away, home), False))
        P = preds[r.game_key]
        is_home = codes.get(r.team_code) == home
        if r.kind == "winner":
            p = P.p_win(is_home)
        elif r.kind == "spread" and r.line == r.line and r.team_code in codes:
            p = P.p_margin_over(r.line, is_home)
        elif r.kind == "total" and r.line == r.line:
            p = P.p_total_over(r.line)
        elif r.kind == "team_total" and r.line == r.line and r.team_code in codes:
            p = P.p_team_over(r.line, is_home)
        else:
            continue
        pm[i] = p if r.side == "yes" else 1 - p
    out = d.copy()
    out["p_model"] = pm
    out.attrs["teams"], out.attrs["preds"] = teams, preds
    return out


def add_estimates(d: pd.DataFrame, cfg: dict, categories: dict | None = None) -> pd.DataFrame:
    """Adds p_est, p_se, source, fee, ev (at the ask), ev_low (at p_est - z x se), edge and Kelly columns."""
    out = d.copy()
    p, se, src = out.mid.values.copy(), np.full(len(out), 0.03), np.array(["price"] * len(out), dtype=object)
    how = {}
    for (sport, kind), idx in out.groupby(["sport", "kind"]).indices.items():
        sub = out.iloc[idx]
        if kind in ("winner", "spread", "total", "team_total"):
            p0, s0, text = stack.estimate(sport, kind, 0.5, 0.5)
            how[(sport, kind)] = text
            if "blend" in text:                                  # an earned blend: apply it row by row
                for j, r in zip(idx, sub.itertuples()):
                    yes_mid = r.mid if r.side == "yes" else 1 - r.mid
                    pmod = (r.p_model if r.side == "yes" else 1 - r.p_model) if r.p_model == r.p_model else None
                    py, sy, _ = stack.estimate(sport, kind, yes_mid, pmod)
                    p[j], se[j], src[j] = (py if r.side == "yes" else 1 - py), sy, "blend"
        elif kind == "prop":
            how[(sport, kind)] = "Kalshi's price (player props: every projection model tested here lost to the price; FINDINGS 2, 17, 19, 21)"
        else:
            how[(sport, kind)] = "Kalshi's price (no tested model for this market type)"
    out["p_est"], out["p_se"], out["source"] = p, se, src
    # graded categories (unders only): the price paid plus the category's measured hit-minus-price gap
    if categories is not None and "cats" in out:
        lab = {sp: botlegs.labels(sp) for sp in ("NBA", "NFL")}
        for j, r in enumerate(out.itertuples()):
            for name in (r.cats or []):
                c = lab.get(r.sport, {}).get(name)
                if c and c["label"] == "PREFER" and c.get("gap") is not None and r.side == "no" and botlegs.BANDS[0][0] <= r.ask < botlegs.BANDS[-1][1]:
                    # only inside the price range the category was measured on (20-85c): no extrapolating to long shots
                    band = next((b for b in (c.get("gap_bands") or []) if b[0] <= r.ask < b[1]), None)          # the gap measured at this price level
                    gap, se = (band[2], band[4] if len(band) > 4 else c["gap_se"]) if band else (c["gap"], c["gap_se"])
                    out.iat[j, out.columns.get_loc("p_est")] = min(r.ask + gap, 0.99)
                    out.iat[j, out.columns.get_loc("p_se")] = max(se, c["gap_se"])
                    out.iat[j, out.columns.get_loc("source")] = f"category: {name}"
    out["fee"] = M.TAKER * out.ask * (1 - out.ask)
    out["ev"] = out.p_est * (1 - out.ask) - (1 - out.p_est) * out.ask - out.fee
    p_low = (out.p_est - cfg["z_edge"] * out.p_se).clip(lower=0)
    out["ev_low"] = p_low * (1 - out.ask) - (1 - p_low) * out.ask - out.fee
    out["ret"], out["ret_low"] = out.ev / (out.ask + out.fee), out.ev_low / (out.ask + out.fee)
    out["edge"] = out.p_est - (out.ask + out.fee)
    out["model_gap"] = out.p_model - out.mid                       # evidence only
    out.attrs.update(d.attrs)
    out.attrs["how"] = how
    return out
