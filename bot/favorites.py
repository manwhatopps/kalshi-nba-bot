"""The "heavy favorites" strategy, tested on Kalshi's own settled markets.

Idea under test: stack many legs that each cost >= 90c (YES on near-certainties, or NO on long
shots) into one combo, e.g. 50 legs, for a big payout.

Every settled market gives two possible legs at the last pre-game quote:
    YES at the ask,  NO at (1 - bid).
We measure (1) leg calibration — do 90c+ legs win more or less often than they cost, by price,
side and market type, net of Kalshi's fee — and (2) whole cards, built night by night with only
information available before that night (walk-forward), scored on what really happened.
Payouts assume the combo costs the product of its leg prices (no extra maker margin), which is
the BEST case: published research on 8.9M Kalshi combos found makers keep ~2-3c on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalshi import fee

MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
BUCKETS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0]


def slate_date(game: str):
    try:
        return pd.Timestamp(2000 + int(game[:2]), MONTHS[game[2:5]], int(game[5:7]))
    except Exception:
        return pd.NaT


def legs_table(hist: pd.DataFrame, when: str = "pre", max_spread: float = 0.15, basis: str = "taker") -> pd.DataFrame:
    """basis='taker': pay the quote (YES at ask, NO at 1-bid).  basis='mid': the midpoint, i.e. what a
    leg is 'worth' to a combo market maker who prices off fair value rather than the posted quote."""
    df = hist[hist.result.isin(["yes", "no"])].copy()
    bid, ask = df[f"{when}_bid"], df[f"{when}_ask"]
    ok = bid.notna() & ask.notna() & (ask > bid) & ((ask - bid) <= max_spread)
    df = df[ok]
    bid, ask = df[f"{when}_bid"], df[f"{when}_ask"]
    if basis == "mid":
        mid = (bid + ask) / 2
        yes = df.assign(side="yes", cost=mid, win=(df.result == "yes"))
        no = df.assign(side="no", cost=1 - mid, win=(df.result == "no"))
    else:
        yes = df[ask < 1].assign(side="yes", cost=ask[ask < 1], win=(df.result[ask < 1] == "yes"))
        no = df[bid > 0].assign(side="no", cost=1 - bid[bid > 0], win=(df.result[bid > 0] == "no"))
    legs = pd.concat([yes, no], ignore_index=True)
    legs["win"] = legs.win.astype(float)
    legs["date"] = legs.game.map(slate_date)
    legs["net"] = legs.win - legs.cost - legs.cost.map(fee)          # profit per contract after fee
    return legs.dropna(subset=["date"])


def _cluster_se(e: pd.Series, groups: pd.Series) -> float:
    s = e.groupby(groups).sum()
    return float(np.sqrt((s ** 2).sum()) / max(len(e), 1))


def calibration(legs: pd.DataFrame, by=None) -> pd.DataFrame:
    legs = legs.assign(bucket=pd.cut(legs.cost, BUCKETS, right=False))
    keys = (list(by) if by else []) + ["bucket"]
    rows = []
    for k, g in legs.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        e = g.win - g.cost
        rows.append(dict(zip(keys, k)) | dict(
            n=len(g), games=g.game.nunique(), price=g.cost.mean(), win=g.win.mean(), edge=e.mean(),
            se=_cluster_se(e - e.mean(), g.game), fee=g.cost.map(fee).mean(), net=g.net.mean(),
            roi=g.net.sum() / g.cost.sum()))
    out = pd.DataFrame(rows)
    out["t"] = out.net / out.se.replace(0, np.nan)
    return out


class WalkForward:
    """p_hat for a leg from PRIOR nights only: shrunken win rate of its (series, side, price bucket)."""

    def __init__(self, kappa: float = 300):
        self.kappa = kappa
        self.stats = {}

    def fit(self, past: pd.DataFrame):
        b = pd.cut(past.cost, BUCKETS, right=False)
        g = past.groupby([past.series, past.side, b], observed=True)
        self.stats = {k: (v.win.sum(), len(v), v.cost.mean()) for k, v in g}
        return self

    def predict(self, legs: pd.DataFrame) -> np.ndarray:
        b = pd.cut(legs.cost, BUCKETS, right=False)
        out = []
        for s, side, bk, c in zip(legs.series, legs.side, b, legs.cost):
            w, n, mc = self.stats.get((s, side, bk), (0, 0, c))
            out.append(c + (w - n * mc) / (n + self.kappa))   # cost + shrunken average edge
        return np.array(out)


def _pick(cands: pd.DataFrame, n: int, per_game: int, rng=None) -> pd.DataFrame | None:
    """Greedy pick in the given order (or random order): one leg per event, per_game cap."""
    order = rng.permutation(len(cands)) if rng is not None else range(len(cands))
    ev, gm, keep = set(), {}, []
    events, games = cands.event_ticker.values, cands.game.values
    for i in order:
        if events[i] in ev or gm.get(games[i], 0) >= per_game:
            continue
        ev.add(events[i])
        gm[games[i]] = gm.get(games[i], 0) + 1
        keep.append(i)
        if len(keep) == n:
            return cands.iloc[keep]
    return None


def backtest_cards(legs: pd.DataFrame, n_legs: int, pmin: float, pmax: float = 0.995, per_game: int = 8,
                   mode: str = "top", samples: int = 200, vig: float = 0.0, warmup: int = 7, seed: int = 1):
    """mode: 'top' = highest-priced legs; 'smart' = best walk-forward expected edge; 'random' = sampled cards."""
    rng = np.random.default_rng(seed)
    dates = sorted(legs.date.unique())
    rows = []
    for d in dates[warmup:]:
        day = legs[(legs.date == d) & (legs.cost >= pmin) & (legs.cost <= pmax)]
        if len(day) < n_legs:
            continue
        if mode == "smart":
            past = legs[legs.date < d]
            day = day.assign(p_hat=WalkForward().fit(past).predict(day))
            day = day[day.p_hat - day.cost - day.cost.map(fee) > 0]
            day = day.sort_values("p_hat", ascending=False)
        elif mode == "top":
            day = day.sort_values("cost", ascending=False)
        cards = []
        if mode == "random":
            for _ in range(samples):
                c = _pick(day, n_legs, per_game, rng)
                if c is not None:
                    cards.append(c)
        else:
            c = _pick(day, n_legs, per_game)
            if c is not None:
                cards.append(c)
        for c in cards:
            price = float(np.prod(c.cost.values)) * (1 + vig)
            price_f = price + fee(min(price, 0.999))
            hit = float(c.win.all())
            rows.append(dict(date=d, hit=hit, legs_hit=c.win.sum(), price=price, mult=1 / price_f,
                             ret=hit / price_f - 1, indep=float(np.prod(c.win.mean() ** 0 * c.cost.values))))
    r = pd.DataFrame(rows)
    if r.empty:
        return None
    nightly = r.groupby("date").agg(hit=("hit", "mean"), ret=("ret", "mean"), price=("price", "mean"),
                                    legs_hit=("legs_hit", "mean"), mult=("mult", "mean"))
    return dict(nights=len(nightly), hit_rate=nightly.hit.mean(), implied=nightly.price.mean(),
                avg_mult=nightly.mult.mean(), roi=nightly.ret.mean(),
                roi_se=nightly.ret.std() / np.sqrt(len(nightly)) if len(nightly) > 1 else np.nan,
                avg_legs_hit=nightly.legs_hit.mean(), nightly=nightly)


def sweep(legs: pd.DataFrame, mode: str, sizes=(2, 3, 5, 10, 20, 30, 50), pmins=(0.80, 0.90, 0.95, 0.97),
          **kw) -> pd.DataFrame:
    rows = []
    for pmin in pmins:
        for n in sizes:
            r = backtest_cards(legs, n, pmin, mode=mode, **kw)
            if r:
                rows.append(dict(min_price=pmin, legs=n, nights=r["nights"], hit_rate=r["hit_rate"],
                                 needed=r["implied"], avg_legs_hit=r["avg_legs_hit"], payout_x=r["avg_mult"],
                                 roi=r["roi"], roi_se=r["roi_se"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tonight's card (live markets)
def live_legs(k, league: str = "NBA", hours_ahead: float = 30, include_live: bool = False,
              max_spread: float = 0.15) -> pd.DataFrame:
    """Every open market of the league as YES/NO legs at the current quote (pre-game by default)."""
    from datetime import datetime, timedelta, timezone
    from .history import NBA_SERIES, NFL_SERIES, START_OFFSET
    now = datetime.now(timezone.utc)
    rows = []
    for s in (NBA_SERIES if league == "NBA" else NFL_SERIES):
        try:
            ms = k.markets(s)
        except Exception as e:
            print(f"  ! {s}: {e}")
            continue
        for m in ms:
            occ = m.get("occurrence_datetime") or m.get("expected_expiration_time")
            if not occ or "-" not in m["event_ticker"]:
                continue
            tip = datetime.fromisoformat(occ.replace("Z", "+00:00")) - START_OFFSET[league]
            live = tip < now + timedelta(minutes=10)
            if (live and not include_live) or tip > now + timedelta(hours=hours_ahead):
                continue
            try:
                bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
            except ValueError:
                continue
            if not (0 <= bid < ask <= 1) or ask - bid > max_spread:
                continue
            base = dict(ticker=m["ticker"], event_ticker=m["event_ticker"], series=s, title=m.get("title", ""),
                        game=m["event_ticker"].split("-")[1], tipoff=tip, live=live)
            if ask < 1:
                rows.append(base | dict(side="yes", cost=ask))
            if bid > 0:
                rows.append(base | dict(side="no", cost=1 - bid))
    return pd.DataFrame(rows)


def build_card(legs_now: pd.DataFrame, hist_legs: pd.DataFrame, n_legs: int = 50, pmin: float = 0.90,
               pmax: float = 0.995, per_game: int = 8, smart: bool = True) -> pd.DataFrame | None:
    c = legs_now[(legs_now.cost >= pmin) & (legs_now.cost <= pmax)].copy()
    if c.empty:
        return None
    c["p_hat"] = WalkForward().fit(hist_legs).predict(c) if len(hist_legs) else c.cost
    c["net_edge"] = c.p_hat - c.cost - c.cost.map(fee)
    c = c.sort_values("net_edge" if smart else "cost", ascending=False)
    return _pick(c, min(n_legs, c.event_ticker.nunique()), per_game)
