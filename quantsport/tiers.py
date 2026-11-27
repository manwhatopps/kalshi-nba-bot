"""The four investment tiers.

  CORE          single positions the logical chain accepted (built in pipeline.py from chain decisions)
  SMALL         two legs, fair payout 1.5x-3x
  DIVERSIFIED   several legs, fair payout near 5x / 10x / 25x
  DEGEN         long shots near 100x / 250x / 500x / 1,000x on a small fixed budget

Card rules (hard, checked again by the independent auditor in risk.py):
  * one leg per game on a card; a same-game card needs an explicit override and is labelled as such
  * no game on two open cards; no player twice on a card
  * every leg must be combinable under Kalshi's own collection rules (winners are YES-only, props take either side)
  * a card is bought only if the quote pays at least `min_keep` x its fair payout (1.0 = no negative expected value
    for SMALL and DIVERSIFIED; 0.90 for DEGEN, which is entertainment money and says so)

Fair payout is 1 / (chance every leg hits). Legs in different games are treated as independent, which is what the
data shows (real combos with one leg per game hit 1.01x the product of their leg prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mathx as M
from . import risk

TIER_ORDER = ("small", "diversified", "degen")


def leg_pool(e: pd.DataFrame, cfg: dict, rules: dict) -> pd.DataFrame:
    """Sides that may go on a card, best first: tight books, mid prices, legal in a combo, not in an 'avoid' category.
    Props are taken as unders only (overs lose 6-11% pre-game, unders about 2%)."""
    lo, hi = cfg["leg_price"]
    p = e[(e.ask >= lo) & (e.ask <= hi) & (e.spread <= cfg["max_spread"]) & e.kind.isin(["winner", "spread", "total", "team_total", "prop"])].copy()
    p = p[(p.kind != "prop") | (p.side == "no")]
    if rules:
        ok = [ev in rules and not (rules[ev][0] and s != "yes") for ev, s in zip(p.event_ticker, p.side)]
        p = p[ok]
    if "why" in p:                                                   # no card leg on a player listed out, doubtful or questionable
        from bot.nba_injuries import blocks
        p = p[[not blocks(w) for w in p.why]]
    if "pref" in p:
        p = p[p.pref.fillna(0) >= 0]
    else:
        p["pref"] = 0.0
    p["quality"] = p.ret + 0.02 * p.pref.fillna(0) - 0.5 * p.spread          # least-bad expected return, preferred categories, tight books
    return p.sort_values("quality", ascending=False)


def _card(tier: str, target: float, legs: pd.DataFrame, cfg: dict, n: int, override: int = 0) -> dict:
    ps = legs.p_est.values
    pj = M.joint_independent(ps)
    fair = M.fair_multiple(pj)
    keep = cfg["min_keep"][tier]
    need = fair * keep
    stake = float(cfg["card_stake"][tier])
    L = [dict(ticker=r.ticker, event_ticker=r.event_ticker, side=r.side, title=r.title, sport=r.sport, kind=r.kind, game_key=r.game_key,
              player=(f"{r.sport}:{r.player}" if r.player else ""), team=(f"{r.sport}:{r.team_code}" if r.team_code else ""),
              ask=float(r.ask), mid=float(r.mid), p=float(r.p_est), p_model=(float(r.p_model) if r.p_model == r.p_model else None),
              start=r.start.isoformat(), source=r.source, why=list(getattr(r, "why", None) or [])) for r in legs.itertuples()]
    return dict(card_id=f"{tier[:3].upper()}{n}", tier=tier, target=float(target), stake=stake, legs=L, n_legs=len(L), p_joint=pj, fair_mult=fair,
                need_mult=need, ev_at_need=M.combo_ev(pj, need), max_loss=stake, max_payout=stake * need, p_lose=1 - pj,
                games=sorted({l["game_key"] for l in L}), teams=sorted({l["team"] for l in L if l["team"]}),
                players=sorted({l["player"] for l in L if l["player"]}), sports=sorted({l["sport"] for l in L}),
                expires=min(l["start"] for l in L), correlation="legs are in different games: treated as independent (measured 1.01x)"
                if override <= 1 else f"SAME-GAME OVERRIDE: up to {override} legs per game; same-game legs can move together",
                override_same_game=override, kind="combo",
                payload=dict(selected_markets=[dict(market_ticker=l["ticker"], event_ticker=l["event_ticker"], side=l["side"]) for l in L]))


def _take(pool: pd.DataFrame, used_games: set, target: float, max_legs: int = 20, per_game: int = 1) -> pd.DataFrame | None:
    """Best available legs, one per game, until the fair payout reaches the target; the last leg is chosen so the card
    does not overshoot (overshooting gives away hit chance for payout the target did not ask for)."""
    thr, price, keep, games, players = 1 / target, 1.0, [], {}, set()
    rows = pool[~pool.game_key.isin(used_games)]
    best = rows.drop_duplicates("game_key") if per_game == 1 else rows
    cand = list(best.itertuples())
    for r in cand:
        if games.get(r.game_key, 0) >= per_game or (r.player and r.player in players):
            continue
        if price * 0.5 <= thr:
            last = [x for x in rows.itertuples() if games.get(x.game_key, 0) < per_game and not (x.player and x.player in players)
                    and x.Index not in [k.Index for k in keep] and price * x.p_est <= thr]
            if last:
                keep.append(max(last, key=lambda x: x.p_est))
                return pool.loc[[k.Index for k in keep]]
        keep.append(r)
        games[r.game_key] = games.get(r.game_key, 0) + 1
        if r.player:
            players.add(r.player)
        price *= r.p_est
        if price <= thr:
            return pool.loc[[k.Index for k in keep]]
        if len(keep) >= max_legs:
            break
    return None


def build(e: pd.DataFrame, cfg: dict, st: dict, reg: risk.Registry, rules: dict) -> dict:
    """Cards for the three card tiers, each passed through the auditor. Returns {tier: [cards]} with a status on each:
    PENDING QUOTE (may be bought if the real quote meets the walk-away line), VETO, or a NO CARD note for the tier."""
    pool = leg_pool(e, cfg, rules)
    out, notes = {t: [] for t in TIER_ORDER}, {}
    work = risk.Registry()
    work.by, work.card_games, work.total = {k: v.copy() for k, v in reg.by.items()}, dict(reg.card_games), reg.total
    work.pending = reg.pending.copy()
    used = set(work.games_on_cards()) if not cfg["reuse_game_across_cards"] else set()
    def allowed(tier):       # cards this tier may still buy inside its weekly allowance
        left = cfg["weekly_spend"][tier] * st["bankroll"] - st.get("week_spend", {}).get(tier, 0.0)
        pot = cfg["budget"][tier] * st["start"] + st.get("tier_pnl", {}).get(tier, 0.0) - reg.by["strategy"][tier]
        return max(int(min(left, pot) // cfg["card_stake"][tier] + 1e-9), 0)
    import datetime as _dt
    wk = _dt.date.today().isocalendar()[1]
    rot = lambda xs: list(xs[wk % len(xs):]) + list(xs[:wk % len(xs)])          # rotate targets week by week
    plan = {"small": [None] * allowed("small"), "diversified": rot(list(cfg["diversified_targets"]))[:allowed("diversified")],
            "degen": rot(list(cfg["degen_targets"]))[:allowed("degen")]}
    for t_ in TIER_ORDER:
        if not plan[t_]:
            notes[t_] = f"NO {t_.upper()} CARD: this week's allowance is spent or the tier's pot is empty"
    per_game = int(cfg["same_game_per_card"])
    for tier in TIER_ORDER:
        n = 0
        for target in plan[tier]:
            if st["scale"] <= 0:
                notes[tier] = "trading is stopped by a loss limit"
                break
            if tier == "small":
                lo, hi = cfg["small_range"]
                want = lo + (hi - lo) * (0.2 + 0.6 * (n / max(len(plan[tier]) - 1, 1)))        # spread the cards across the range
                a = pool[~pool.game_key.isin(used) & (pool.p_est >= 0.55)].drop_duplicates("game_key")
                legs, gap = None, 9e9
                pv = a.p_est.values
                for i in range(len(a)):
                    for j in range(i + 1, len(a)):
                        m = 1 / (pv[i] * pv[j])
                        if lo <= m <= hi and abs(m - want) < gap:
                            legs, gap = a.iloc[[i, j]], abs(m - want)
                target = 0.0 if legs is None else round(1 / float(np.prod(legs.p_est)), 1)
            else:
                legs = _take(pool, used, float(target), per_game=per_game)
            if legs is None:
                notes[tier] = f"NO {tier.upper()} CARD beyond {n}: not enough eligible legs in different games" if n else \
                    f"NO {tier.upper()} CARD: not enough eligible legs in different games"
                break
            n += 1
            card = _card(tier, target, legs, cfg, n, override=per_game if per_game > 1 else 0)
            from bot import slips
            card["problems"] = slips.legal([dict(ticker=l["ticker"], event_ticker=l["event_ticker"], side=l["side"]) for l in card["legs"]], rules)
            vetoes = risk.audit(card, work, st, cfg)
            card["vetoes"] = vetoes
            card["status"] = "VETO" if vetoes else "PENDING QUOTE"
            out[tier].append(card)
            if not vetoes:
                work.add(dict(stake=card["stake"], tier=tier, games=card["games"], teams=card["teams"], players=card["players"],
                              sports=card["sports"], kind="combo", ref=card["card_id"], new=True))
                if not cfg["reuse_game_across_cards"]:
                    used |= set(card["games"])
    return dict(cards=out, notes=notes, pool=len(pool), pool_games=int(pool.game_key.nunique()) if len(pool) else 0, registry=work)
