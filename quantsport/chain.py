"""The logical chain: the sixteen steps every candidate goes through, ending in ACCEPT, WAIT, or NO BET.

  1 market and settlement rule      5 availability checks          9 fees, spread, slippage     13 position size
  2 current price                   6 statistical evidence        10 uncertainty                14 hedge available?
  3 order book                      7 our probability             11 correlation with the book  15 decision
  4 sports data                     8 versus the market           12 expected value             16 record

NO BET is the normal result. A candidate is accepted only if its expected value is positive at the CAUTIOUS end of
our probability, after the fee and after walking the real order book, and the independent auditor finds no limit
breached. The thesis keeps facts, model outputs, assumptions and judgments apart, and says what was not checked.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from bot import book as B

from . import mathx as M
from . import risk


def wait_price(p_est: float, p_se: float, z: float) -> float:
    """The highest ask at which the cautious estimate still has positive expected value (0 if none)."""
    p_low = max(p_est - z * p_se, 0.0)
    lo, hi = 0.0, 0.99
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_low - (mid + M.fee(mid)) > 0:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def availability(c) -> list[str]:
    """What the system does and does not know about who will play. Stated, never assumed."""
    notes = []
    if c.get("kind") == "prop":
        if c.get("days_rest") == c.get("days_rest") and c.get("days_rest") is not None:
            notes.append(f"last played {int(c['days_rest'])} day(s) before this game (from the game log)")
        if c.get("weeks_out") == c.get("weeks_out") and c.get("weeks_out") is not None and c.get("weeks_out", 0) >= 3:
            notes.append(f"first game back after about {int(c['weeks_out'])} weeks out (from the game log)")
        inj = [w for w in (c.get("why") or []) if str(w).startswith("injury report:")]
        if inj and not any("not published" in w for w in inj):
            notes += [w + " (official weekly report via nflverse, read today)" for w in inj]
            notes.append("NOT VERIFIED: game-day inactives and the starting lineup, which come out 90 minutes before kickoff. "
                         "Kalshi settles a no-show leg at a fair price, so a scratch does not zero the bet.")
        else:
            notes.append("NOT VERIFIED: tonight's injury report and starting lineup (no timestamped injury feed for this sport or week). "
                         "Kalshi settles a no-show leg at a fair price, so a scratch does not zero the bet.")
    else:
        notes.append("NOT VERIFIED: injuries, quarterback or star availability, weather. The rating model knows results only.")
    return notes


def resting_note(c: dict, step: int = 2) -> str:
    """What history says about waiting with a resting bid instead of paying the ask (`python -m bot resting`)."""
    if c.get("kind") != "prop" or not str(c.get("source", "")).startswith("category: "):
        return ""
    try:
        from bot import legs, resting
        name = c["source"].split("category: ", 1)[1]
        r = resting.lookup(f"{c['sport']}: {name} ({legs.labels(c['sport'])[name]['label']})", step)
    except Exception:
        return ""
    if not r or r.get("ret") is None:
        return ""
    return (f"a resting bid {step}c under the ask was filled before the game {r['fill']:.0%} of the time in this category ({r['orders']:,} orders, both seasons); the filled ones "
            f"returned {r['ret']:+.1%} (+/- {r['se']:.1%}) against {r['taker']:+.1%} for paying the ask at once. Bids fill more often on the bets that go on to lose, "
            "so waiting below the ask is not a proven improvement")


def decide(c: dict, cfg: dict, st: dict, reg: risk.Registry, book: dict | None = None, tier: str = "core") -> dict:
    """Run one candidate (a row of the estimate table as a dict) through the chain."""
    now = datetime.now(timezone.utc)
    z = cfg["z_edge"]
    steps, facts, model_out, assume, judge, support, against = [], [], [], [], [], [], []
    # 1-2: market, rule, price
    steps.append(f"1 market: {c['title']} [{c['ticker']}], buy {c['side'].upper()}. Rule: {c.get('rules') or 'MISSING'}")
    age = (now - c["scanned"]).total_seconds() if c.get("scanned") is not None else 1e9
    steps.append(f"2 price: ask {c['ask']:.2f}, bid {c['bid']:.2f}, spread {c['spread']:.2f}, quote {age:.0f}s old")
    facts.append(f"Kalshi asks {c['ask'] * 100:.0f}c for {c['side'].upper()}; the market's own implied chance (mid) is {c['mid']:.1%}")
    # 3: order book
    stake_try = cfg["kelly_cap"] * st["bankroll"]
    if book is not None:
        limit = min(round(c["ask"] + 0.02, 2), 0.99)
        fill = B.buy(book, c["side"], stake_try, limit)
        depth, avg = fill["offered_within_limit"] * c["ask"], (fill["avg"] if fill["contracts"] else c["ask"])
        steps.append(f"3 book: {fill['offered_within_limit']:.0f} contracts offered within 2c of the ask (about ${depth:,.0f}); average fill {avg:.3f}")
    else:
        depth, avg = c.get("ask_size", 0) * c["ask"], c["ask"]
        steps.append(f"3 book: top of book only, {c.get('ask_size', 0):.0f} contracts shown at the ask")
    # 4-6: data, availability, evidence
    steps.append("4 sports data: " + (f"rating model output available ({c['p_model']:.1%})" if c.get("p_model") == c.get("p_model") and c.get("p_model") is not None
                                      else "no model output for this market"))
    avail = availability(c)
    steps.append("5 availability: " + " | ".join(avail))
    if c.get("p_model") == c.get("p_model") and c.get("p_model") is not None:
        gap = c["p_model"] - c["mid"]
        model_out.append(f"rating model: {c['p_model']:.1%} (market {c['mid']:.1%}, difference {gap:+.1%})")
        (support if gap > 0.03 else against if gap < -0.03 else model_out).append(
            f"the rating model {'agrees with' if abs(gap) <= 0.03 else ('is higher than' if gap > 0 else 'is lower than')} the price by {abs(gap):.1%}"
            + (" (this model has not beaten the price on settled markets, so it is context only)" if c.get("source") == "price" else ""))
    for w in (c.get("why") or []):
        (support if w.startswith(("PREFER", "lean prefer")) else against if w.startswith(("AVOID", "lean avoid")) else facts).append(w)
    steps.append(f"6 evidence: {len(support)} supporting, {len(against)} contradicting, {len(facts)} facts")
    # 7-8: our probability vs the market
    p, se = float(c["p_est"]), float(c["p_se"])
    how = c.get("how") or c.get("source", "price")
    model_out.append(f"our probability {p:.1%} +/- {se:.1%} ({how})")
    steps.append(f"7 our probability: {p:.1%}, standard error {se:.1%}")
    steps.append(f"8 versus market: ours {p:.1%}, market mid {c['mid']:.1%}, ask {c['ask']:.1%}")
    # 9-10, 12: costs, uncertainty, EV
    fee = M.fee(avg)
    ev, p_low = M.ev(p, avg, fee), max(p - z * se, 0.0)
    ev_low = M.ev(p_low, avg, fee)
    steps.append(f"9 costs: fee {fee * 100:.2f}c a contract, spread {c['spread'] * 100:.0f}c, slippage {max(avg - c['ask'], 0) * 100:.1f}c; break-even chance {M.break_even(avg, fee):.1%}")
    steps.append(f"10 uncertainty: cautious estimate {p_low:.1%} ({z:g} standard errors below)")
    # 11: correlation with the book
    held = reg.by["event"].get(c["game_key"], 0.0)
    team_held = reg.by["team"].get(f"{c['sport']}:{c.get('team_code')}", 0.0) if c.get("team_code") else 0.0
    steps.append(f"11 portfolio: ${held:.2f} already on this game, ${team_held:.2f} on this team, ${reg.total:.2f} open in total")
    steps.append(f"12 expected value: {ev:+.3f} per contract at our estimate ({M.expected_return(p, avg, fee):+.1%} per $1); {ev_low:+.3f} at the cautious estimate")
    # 13: size
    f_full = M.kelly(p, avg, fee)
    f = M.kelly_sized(p, avg, fee, cfg["kelly_fraction"], cfg["kelly_cap"], se, z) * st["scale"]
    stake = round(f * st["bankroll"], 2)
    n, cost = M.contracts_for(stake, avg) if stake > 0 else (0, 0.0)
    steps.append(f"13 size: full Kelly {f_full:.1%}; {cfg['kelly_fraction']:g} x Kelly on the cautious estimate, capped at {cfg['kelly_cap']:.0%}, "
                 f"risk scale {st['scale']:g} -> {f:.2%} of bankroll = ${stake:.2f} ({n} contracts)")
    # 14: hedge
    opp = 1 - c["bid"]
    steps.append(f"14 hedge: the other side costs {opp * 100:.0f}c; buying both costs {(c['ask'] + opp) * 100:.0f}c plus two fees for a certain $1, "
                 "so a hedge only makes sense after the price has moved in our favour")
    # 15: decision, with the auditor
    proposal = dict(stake=cost, tier=tier, kind="single", games=[c["game_key"]], teams=[f"{c['sport']}:{c['team_code']}"] if c.get("team_code") else [],
                    players=[f"{c['sport']}:{c['player']}"] if c.get("player") else [], sport=c["sport"], ev_low=ev_low, spread=c["spread"],
                    depth_dollars=depth, quote_age_s=age, rules=c.get("rules"))
    vetoes = risk.audit(proposal, reg, st, cfg) if ev_low > cfg["min_ev"] and n > 0 else []
    wp = wait_price(p, se, z)
    if ev_low > cfg["min_ev"] and n > 0 and not vetoes:
        decision, why = "ACCEPT", f"positive expected value even at the cautious estimate ({ev_low:+.3f} per contract) and no limit breached"
    elif ev_low > cfg["min_ev"] and vetoes:
        decision, why = "VETO", "the risk auditor refused it: " + "; ".join(vetoes)
    elif ev > 0 and wp >= 0.05:
        decision, why = "WAIT", f"worth buying only at {wp * 100:.0f}c or less (now {c['ask'] * 100:.0f}c): positive at our estimate, not at the cautious one"
    else:
        decision, why = "NO BET", (f"expected value {ev:+.3f} per contract: the price is {'our estimate' if c.get('source') == 'price' else 'above fair'}"
                                   f"{', so the fee and the spread are a pure cost' if c.get('source') == 'price' else ''}")
    judge.append(why)
    if decision == "WAIT":
        rest = resting_note(c)
        if rest:
            judge.append(rest)
    assume += ["the standard error covers model error as well as sampling error", "the quote can still be had at the size shown"]
    fails = ["a late scratch or minutes limit" if c.get("kind") == "prop" else "an injury or weather change the model cannot see",
             "the game script departs from the usual (blowout, overtime)", "our estimate is the market's own price, so being 'right' still loses the fee"
             if c.get("source") == "price" else "the measured edge was a product of looking at many rules"]
    steps.append(f"15 decision: {decision}. {why}")
    steps.append("16 recorded with this evidence and model version")
    return dict(ticker=c["ticker"], side=c["side"], sport=c["sport"], kind=c["kind"], game_key=c["game_key"], title=c["title"], tier=tier,
                decision=decision, reason=why, p_est=p, p_se=se, p_market=float(c["mid"]), p_model=(float(c["p_model"]) if c.get("p_model") == c.get("p_model") and c.get("p_model") is not None else None),
                ask=float(c["ask"]), fee=fee, ev=ev, ev_low=ev_low, ret=M.expected_return(p, avg, fee), kelly=f_full, stake=cost if decision == "ACCEPT" else 0.0,
                contracts=n if decision == "ACCEPT" else 0, wait_price=wp, source=c.get("source", "price"), vetoes=vetoes,
                proposal=proposal, max_loss=cost, pays=float(n),
                thesis=dict(facts=facts, model_outputs=model_out, assumptions=assume, judgments=judge, supporting=support, contradicting=against,
                            availability=avail, failure_scenarios=fails, steps=steps,
                            hedge=f"other side at {opp * 100:.0f}c", correlated=f"${held:.2f} on this game, ${team_held:.2f} on this team"))
