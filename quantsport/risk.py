"""Portfolio state, the exposure registry, hard limits, the independent auditor, and portfolio simulation.

The auditor is deliberately separate from the strategy builder: it sees only a proposed position, the book of open
positions and the configuration, and returns the reasons it must be refused. The builder cannot switch a check off;
a limit can only be changed in data/quantsport.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import mathx as M
from . import store


# ---------------------------------------------------------------- state
def state(con, cfg: dict) -> dict:
    """Bankroll, cash, exposure, realized P&L today and this week, drawdown, and the factor new risk is scaled by."""
    pos = store.frame(con, "SELECT * FROM positions")
    start = float(cfg["bankroll"])
    closed = pos[pos.status != "open"] if len(pos) else pos
    realized = float(closed.pnl.fillna(0).sum()) if len(closed) else 0.0
    open_ = pos[pos.status == "open"] if len(pos) else pos
    exposure = float(open_.stake.sum()) if len(open_) else 0.0
    bankroll = start + realized
    now = datetime.now(timezone.utc)

    def since(days):
        if not len(closed):
            return 0.0
        t = pd.to_datetime(closed.closed, utc=True)
        return float(closed.pnl[t >= now - timedelta(days=days)].fillna(0).sum())
    eq = store.frame(con, "SELECT * FROM equity ORDER BY ts")
    peak = max([start, bankroll] + ([float(eq.bankroll.max())] if len(eq) else []))
    dd = max((peak - bankroll) / peak, 0.0) if peak > 0 else 0.0
    scale = M.drawdown_scale(dd, [tuple(s) for s in cfg["drawdown_steps"]])
    stops = []
    if since(1) <= -cfg["max_daily_loss"] * start:
        stops.append(f"daily loss limit reached ({since(1):+.2f})")
    if since(7) <= -cfg["max_weekly_loss"] * start:
        stops.append(f"weekly loss limit reached ({since(7):+.2f})")
    if dd >= cfg["max_drawdown"]:
        stops.append(f"maximum drawdown reached ({dd:.0%})")
    week_spend = {}
    if len(pos):
        recent = pos[pd.to_datetime(pos.opened, utc=True) >= now - timedelta(days=7)]
        week_spend = {t: float(v) for t, v in recent.groupby("tier").stake.sum().items()}
    return dict(start=start, bankroll=bankroll, cash=bankroll - exposure, exposure=exposure, realized=realized, day=since(1), week=since(7),
                week_spend=week_spend, tier_pnl=({t: float(v) for t, v in closed.groupby("tier").pnl.sum().items()} if len(closed) else {}),
                peak=peak, drawdown=dd, scale=0.0 if stops else scale, stops=stops, open=len(open_), settled=len(closed))


def calibration_guard(con, cfg: dict) -> tuple[float, str]:
    """If our own estimates have scored worse than Kalshi's price on settled decisions, halve new risk."""
    d = store.frame(con, "SELECT p_est, p_market, result FROM decisions WHERE result IN ('won','lost')")
    g = cfg["calibration_guard"]
    if len(d) < g["min_settled"]:
        return 1.0, f"calibration guard idle: {len(d)} settled decisions, needs {g['min_settled']}"
    y = (d.result == "won").astype(float)
    ours, mkt = float(((d.p_est - y) ** 2).mean()), float(((d.p_market - y) ** 2).mean())
    if ours - mkt > g["max_brier_excess"]:
        return 0.5, f"our estimates score worse than the price (Brier {ours:.4f} vs {mkt:.4f}): new risk halved"
    return 1.0, f"calibration fine (Brier {ours:.4f} vs price {mkt:.4f} on {len(d)} settled)"


# ---------------------------------------------------------------- exposure registry
class Registry:
    """Every open dollar, indexed by game, team, player, sport and strategy."""

    def __init__(self, con=None):
        self.by = {k: defaultdict(float) for k in ("event", "team", "player", "sport", "strategy")}
        self.card_games: dict[str, set] = {}
        self.total = 0.0
        self.pending = defaultdict(float)          # stakes proposed in this run, by tier (counted against weekly allowances)
        if con is not None:
            for r in store.frame(con, "SELECT * FROM positions WHERE status='open'").itertuples():
                self.add(dict(stake=r.stake, tier=r.tier, sport=r.sport, games=json.loads(r.games or "[]"), teams=json.loads(r.teams or "[]"),
                              players=json.loads(r.players or "[]"), kind=r.kind, ref=r.ref))

    def add(self, p: dict) -> None:
        st = float(p["stake"])
        self.total += st
        for g in p.get("games", []):
            self.by["event"][g] += st
        for t in p.get("teams", []):
            self.by["team"][t] += st
        for pl in p.get("players", []):
            self.by["player"][pl] += st
        for sp in set(p.get("sports") or [p.get("sport")]):
            if sp:
                self.by["sport"][sp] += st / max(len(set(p.get("sports") or [p.get("sport")])), 1)
        self.by["strategy"][p.get("tier", "core")] += st
        if p.get("new"):
            self.pending[p.get("tier", "core")] += st
        if p.get("kind") == "combo":
            self.card_games[p.get("ref", f"card{len(self.card_games)}")] = set(p.get("games", []))

    def games_on_cards(self) -> set:
        return set().union(*self.card_games.values()) if self.card_games else set()


# ---------------------------------------------------------------- the independent auditor
def audit(p: dict, reg: Registry, st: dict, cfg: dict) -> list[str]:
    """Reasons this proposed position must be refused. Empty list = it may proceed. `p` needs: stake, tier, kind
    ('single' or 'combo'), games, teams, players, sports; singles also ev_low, spread, depth_dollars, quote_age_s,
    rules; combos also legs (each with game_key) and override_same_game."""
    B, lim, out = st["bankroll"], cfg["limit"], []
    stake = float(p["stake"])
    if st["stops"]:
        out += [f"trading stopped: {s}" for s in st["stops"]]
    if stake <= 0:
        out.append("no stake")
    if stake > st["cash"] + 1e-9:
        out.append(f"stake {stake:.2f} exceeds available cash {st['cash']:.2f}")
    tier = p.get("tier", "core")
    pot = cfg["budget"][tier] * st["start"] + (st.get("tier_pnl", {}).get(tier, 0.0) if tier != "core" else 0.0)      # card pots shrink with losses
    if reg.by["strategy"][tier] + stake > pot + 1e-9:
        out.append(f"{tier} {'budget' if tier == 'core' else 'pot'} cannot cover it ({reg.by['strategy'][tier]:.2f} at risk, {max(pot, 0):.2f} available)")
    cap = cfg.get("weekly_spend", {}).get(tier)
    if cap is not None and st.get("week_spend", {}).get(tier, 0.0) + reg.pending.get(tier, 0.0) + stake > cap * B + 1e-9:
        out.append(f"{tier} weekly allowance spent ({st.get('week_spend', {}).get(tier, 0.0) + reg.pending.get(tier, 0.0):.2f} of {cap * B:.2f} in the last 7 days)")
    for key, items in (("event", p.get("games", [])), ("team", p.get("teams", [])), ("player", p.get("players", []))):
        for it in items:
            if reg.by[key][it] + stake > lim[key] * B + 1e-9:
                out.append(f"{key} limit: {it} would carry {reg.by[key][it] + stake:.2f}, limit {lim[key] * B:.2f}")
    for sp in set(p.get("sports") or [p.get("sport")]):
        if sp and reg.by["sport"][sp] + stake > lim["sport"] * B + 1e-9:
            out.append(f"sport limit: {sp}")
    if p["kind"] == "single":
        if not p.get("rules"):
            out.append("settlement rule text missing")
        if p.get("ev_low", -1) <= cfg["min_ev"]:
            out.append(f"expected value at the cautious estimate is {p.get('ev_low', float('nan')):+.3f} per contract: not positive")
        if p.get("spread", 1) > cfg["max_spread"]:
            out.append(f"bid/ask spread {p.get('spread', 1):.2f} wider than {cfg['max_spread']:.2f}")
        if p.get("depth_dollars", 0) < min(cfg["min_depth_dollars"], stake):
            out.append(f"only {p.get('depth_dollars', 0):.2f} dollars offered inside the limit")
        if p.get("quote_age_s", 1e9) > cfg["quote_max_age_s"]:
            out.append(f"quote is {p.get('quote_age_s', 0):.0f}s old (limit {cfg['quote_max_age_s']}s)")
        if stake > cfg["kelly_cap"] * B + 1e-9:
            out.append("stake above the per-position cap")
    else:
        games = [l["game_key"] for l in p.get("legs", [])]
        most = max((games.count(g) for g in set(games)), default=0)
        allowed = int(p.get("override_same_game") or cfg["same_game_per_card"])
        if most > allowed:
            out.append(f"{most} legs from one game on a card (rule: {allowed}); needs an explicit same-game override")
        if not cfg["reuse_game_across_cards"]:
            clash = set(games) & reg.games_on_cards()
            if clash:
                out.append(f"game already on another open card: {sorted(clash)[0]}")
        pl = [l.get("player") for l in p.get("legs", []) if l.get("player")]
        if len(pl) != len(set(pl)):
            out.append("the same player appears twice on the card")
        if p.get("problems"):
            out.append("Kalshi combo rules: " + "; ".join(p["problems"][:2]))
        if len(p.get("legs", [])) < 2:
            out.append("a combo needs at least 2 legs")
    return out


# ---------------------------------------------------------------- simulation and optimisation
def simulate_portfolio(positions: list[dict], sims: int = 50_000, seed: int = 3, same_game_rho: float = 0.0, same_player_rho: float = 0.35) -> dict:
    """Monte Carlo of the open book. Each position: stake, pays (dollars if it wins), legs [(p, game_key, player)].
    Legs in different games are independent; same-player legs are tied by `same_player_rho` (measured 0.2-0.4)."""
    legs, owner = [], []
    for i, p in enumerate(positions):
        for l in p["legs"]:
            legs.append(l)
            owner.append(i)
    if not legs:
        return dict(expected=0.0, var95=0.0, cvar95=0.0, p_loss=0.0, worst=0.0, best=0.0, pnl=np.zeros(1))
    n = len(legs)
    C = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            if legs[a][2] and legs[a][2] == legs[b][2] and legs[a][1] == legs[b][1]:
                C[a, b] = C[b, a] = same_player_rho
            elif legs[a][1] == legs[b][1]:
                C[a, b] = C[b, a] = same_game_rho
    from scipy.stats import norm
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(n), C, size=sims, method="cholesky")
    hit = z < norm.ppf(np.clip([l[0] for l in legs], 1e-6, 1 - 1e-6))
    pnl = np.zeros(sims)
    owner = np.array(owner)
    for i, p in enumerate(positions):
        win = hit[:, owner == i].all(axis=1)
        pnl += np.where(win, p["pays"] - p["stake"], -p["stake"])
    v, cv = M.var_cvar(pnl)
    return dict(expected=float(pnl.mean()), var95=v, cvar95=cv, p_loss=float((pnl < 0).mean()), worst=float(pnl.min()), best=float(pnl.max()), pnl=pnl)


def kelly_portfolio(p: np.ndarray, cost: np.ndarray, groups: list[str], cap: float, budget: float, group_cap: float, fraction: float = 1.0,
                    sims: int = 4000, seed: int = 5) -> np.ndarray:
    """Growth-optimal stakes (shares of bankroll) for several single bets at once: maximise the average log of
    1 + sum(f_i x return_i) over simulated joint outcomes, subject to per-bet, per-game and total limits.
    Bets on the same game share one hidden outcome so they are not treated as independent. Solved with cvxpy."""
    import cvxpy as cp
    p, cost = np.asarray(p, float), np.asarray(cost, float)
    n = len(p)
    if n == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    u = rng.random((sims, n))
    gid = {g: rng.random(sims) for g in set(groups)}                 # same game -> blend in a shared draw
    for i, g in enumerate(groups):
        u[:, i] = 0.7 * u[:, i] + 0.3 * gid[g] if groups.count(g) > 1 else u[:, i]
    u = (u.argsort(axis=0).argsort(axis=0) + 0.5) / sims                # back to uniform marginals
    win = u < p
    ret = np.where(win, (1 - cost) / cost, -1.0)
    f = cp.Variable(n, nonneg=True)
    cons = [f <= cap, cp.sum(f) <= budget]
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        if len(idx) > 1:
            cons.append(cp.sum(f[idx]) <= group_cap)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.log1p(ret @ f)) / sims), cons)
    try:
        prob.solve()
        x = np.clip(np.asarray(f.value, float), 0, None) if f.value is not None else np.zeros(n)
    except Exception:
        x = np.zeros(n)
    x[(x < 1e-3) | (p <= cost)] = 0.0                                # sampling noise is not an edge
    return x * fraction
