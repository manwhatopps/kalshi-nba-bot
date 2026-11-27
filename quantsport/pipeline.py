"""One daily investment analysis, end to end. The 'departments' are plain functions on one shared data layer:

  market intelligence   scanner.scan                    -> every buyable side, with rules and quotes
  NBA / football quants models.stack.live, estimate     -> rating-model probabilities; leg facts from bot/legs.py
  probability & maths   estimate.add_estimates, mathx   -> our probability, standard error, EV, Kelly
  portfolio manager     risk.state, risk.Registry       -> bankroll, exposure, drawdown, risk scale
  strategy builder      chain.decide (core), tiers.build-> the four tiers
  risk auditor          risk.audit                      -> independent veto on every position
  hedge analyst         bot.hedge                       -> hedge plan for any open card (`python -m bot hedge`)
  performance research  audit.settle, audit.report      -> calibration and P&L by sport, market and tier
  chief investment officer  report()                    -> the written decision

Modes: research (analyse only), paper (also open simulated positions), live_review (also write order tickets that
need a signed, one-use approval before `execution.place_approved` will send anything).
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from bot import config as botcfg
from bot.kalshi import data_client

from . import MODEL_VERSION, chain, config, estimate, risk, scanner, store, tiers
from .models import stack

OUT = botcfg.DATA_DIR / "quantsport"


def _leg_facts(k, e: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Attach the graded leg categories and plain facts (bot/legs.py) to prop unders, where they exist."""
    e = e.copy()
    e["cats"], e["pref"], e["why"] = [[] for _ in range(len(e))], 0.0, [[] for _ in range(len(e))]
    try:
        from bot import degen
        c = degen.candidates(k, hours=min(hours, 30), lo=0.20, hi=0.851, max_spread=0.06)
    except Exception as ex:
        e.attrs["facts_error"] = str(ex)[:120]
        return e
    if c is None or c.empty:
        return e
    idx = {(t, "no"): i for i, (t, s) in enumerate(zip(e.ticker, e.side)) if s == "no"}
    ci, pi, wi = e.columns.get_loc("cats"), e.columns.get_loc("pref"), e.columns.get_loc("why")
    for r in c.itertuples():
        i = idx.get((r.ticker, "no"))
        if i is not None:
            e.iat[i, ci], e.iat[i, pi], e.iat[i, wi] = list(r.cats), float(r.pref), [w for w in str(r.why).split(" | ") if w]
    return e


def run(mode: str | None = None, hours: float | None = None, top: int = 12, verbose: bool = True) -> dict:
    cfg = config.load()
    mode = mode or cfg["mode"]
    hours = hours or cfg["card_horizon_h"]
    k = data_client()
    con = store.connect()
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]
    # market intelligence
    raw = scanner.scan(k, tuple(cfg["sports"]), hours=hours)
    if raw.empty:
        return dict(run_id=run_id, error="no open markets found")
    # analysts + maths
    rt = stack.live(tuple(cfg["sports"]))
    e = estimate.model_probs(raw, rt)
    e = _leg_facts(k, e, hours)
    e = estimate.add_estimates(e, cfg, categories={})
    how = e.attrs.get("how", {})
    # portfolio manager
    st = risk.state(con, cfg)
    guard, guard_note = risk.calibration_guard(con, cfg)
    st["scale"] *= guard
    reg = risk.Registry(con)
    # strategy: core singles through the chain. Full chain for everything that is positive at our estimate, plus the closest misses.
    screen = e[(e.spread <= cfg["max_spread"]) & (e.ask >= 0.05) & (e.ask <= 0.95)]
    pos_ev = screen[screen.ev > 0]
    # a player's ladder (40+, 50+, 60+ ...) is one position, not eight: keep the line with the best cautious return
    n_before = len(pos_ev)
    pos_ev = one_per_ladder(pos_ev)
    ladder_dropped = n_before - len(pos_ev)
    cand = pd.concat([pos_ev, screen[screen.ev <= 0].sort_values("ret", ascending=False).head(top)]).drop_duplicates(["ticker", "side"])
    from bot import book as B
    books = B.orderbooks(k, list(cand.ticker.unique())) if len(cand) else {}
    decisions = []
    for r in cand.to_dict("records"):
        r["how"] = how.get((r["sport"], r["kind"]), r.get("source"))
        r["scanned"] = pd.Timestamp.now(tz="UTC").to_pydatetime() if r["ticker"] in books else r["scanned"]      # the book was just re-read
        d = chain.decide(r, cfg, st, reg, books.get(r["ticker"]))
        if d["decision"] == "ACCEPT":
            reg.add(dict(stake=d["stake"], tier="core", games=[d["game_key"]], teams=d["proposal"]["teams"], players=d["proposal"]["players"],
                         sport=d["sport"], kind="single", ref=d["ticker"], new=True))
        decisions.append(d)
    accepted = [d for d in decisions if d["decision"] == "ACCEPT"]
    # card tiers
    from bot import slips
    try:
        rules = slips.collection_rules(k)
    except Exception:
        rules = {}
    built = tiers.build(e, cfg, st, reg, rules)
    cards = [c for t in tiers.TIER_ORDER for c in built["cards"][t]]
    # portfolio simulation of what would be open after this run
    sim_pos = [dict(stake=d["stake"], pays=d["pays"], legs=[(d["p_est"], d["game_key"], "")]) for d in accepted] + \
              [dict(stake=c["stake"], pays=c["max_payout"], legs=[(l["p"], l["game_key"], l["player"]) for l in c["legs"]]) for c in cards if c["status"] == "PENDING QUOTE"]
    sim = risk.simulate_portfolio(sim_pos) if sim_pos else None
    # record
    store.save_scan(run_id, e)
    ts = store.now()
    for d in decisions:
        store.insert(con, "decisions", dict(run_id=run_id, ts=ts, sport=d["sport"], kind=d["kind"], ticker=d["ticker"], side=d["side"], game_key=d["game_key"],
                                            title=d["title"], tier=d["tier"], decision=d["decision"], p_est=d["p_est"], p_se=d["p_se"], p_market=d["p_market"],
                                            p_model=d["p_model"], ask=d["ask"], fee=d["fee"], ev=d["ev"], ev_low=d["ev_low"], kelly=d["kelly"], stake=d["stake"],
                                            contracts=d["contracts"], wait_price=d["wait_price"], source=d["source"], thesis=d["thesis"], audit=d["vetoes"],
                                            model_version=MODEL_VERSION))
    for c in cards:
        cid = f"{run_id}:{c['card_id']}"
        store.insert(con, "cards", dict(card_id=cid, run_id=run_id, ts=ts, tier=c["tier"], target=c["target"], stake=c["stake"], p_joint=c["p_joint"],
                                        fair_mult=c["fair_mult"], need_mult=c["need_mult"], ev_at_need=c["ev_at_need"], n_legs=c["n_legs"], legs=c["legs"],
                                        games=c["games"], expires=c["expires"], status=c["status"], audit=c["vetoes"]))
    if mode in ("paper", "live_review"):
        for d in accepted:
            store.insert(con, "positions", dict(opened=ts, mode="paper", kind="single", ref=d["ticker"] + ":" + d["side"], tier="core", sport=d["sport"], label=d["title"],
                                                stake=d["stake"], pays=d["pays"], tickers=[d["ticker"] + ":" + d["side"]], games=[d["game_key"]],
                                                teams=d["proposal"]["teams"], players=d["proposal"]["players"], expires=None))
        for c in cards:
            if c["status"] == "PENDING QUOTE":      # paper book: assumed bought at the walk-away payout (no real quote without a key)
                store.insert(con, "positions", dict(opened=ts, mode="paper", kind="combo", ref=f"{run_id}:{c['card_id']}", tier=c["tier"], sport=",".join(c["sports"]),
                                                    label=f"{c['tier']} {c['need_mult']:.1f}x, {c['n_legs']} legs", stake=c["stake"], pays=c["max_payout"],
                                                    tickers=[l["ticker"] + ":" + l["side"] for l in c["legs"]], games=c["games"], teams=c["teams"],
                                                    players=c["players"], expires=c["expires"]))
    st2 = risk.state(con, cfg)
    store.insert(con, "equity", dict(ts=ts, bankroll=st2["bankroll"], open_exposure=st2["exposure"], realized=st2["realized"], peak=st2["peak"], drawdown=st2["drawdown"]))
    store.insert(con, "runs", dict(run_id=run_id, ts=ts, mode=mode, model_version=MODEL_VERSION, scanned=len(e), games=int(e.game_key.nunique()),
                                   accepted=len(accepted), notes=guard_note))
    summary = e.groupby(["sport", "kind"]).agg(sides=("ticker", "size"), games=("game_key", "nunique"), best_ret=("ret", "max")).reset_index()
    out = dict(run_id=run_id, ts=ts, mode=mode, cfg=cfg, state_before=st, state=st2, guard=guard_note, scanned=len(e), games=int(e.game_key.nunique()),
               identified=len(e.attrs.get("teams", {})), summary=summary.to_dict("records"), how={f"{a}:{b}": v for (a, b), v in how.items()},
               decisions=decisions, accepted=len(accepted), ladder_dropped=int(ladder_dropped), cards=cards, tier_notes=built["notes"], pool=built["pool"], pool_games=built["pool_games"],
               sim=({k_: v for k_, v in sim.items() if k_ != "pnl"} if sim else None), errors=raw.attrs.get("errors", []),
               disagreements=_disagreements(e), model_version=MODEL_VERSION)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / f"{run_id}.json").write_text(json.dumps(out, default=str))
    if verbose:
        print(report(out))
    return out


def one_per_ladder(e: pd.DataFrame) -> pd.DataFrame:
    """A player's ladder of lines (40+, 50+, 60+ ... on one stat, one side) is one position: keep the line with the
    best return at the cautious end of our estimate. Game-level markets (no player) pass through untouched."""
    lad = e[e.player != ""]
    best = lad.sort_values("ret_low", ascending=False).drop_duplicates(["sport", "player", "series", "side"])
    return pd.concat([e[e.player == ""], best])


def _disagreements(e: pd.DataFrame, n: int = 8) -> list[dict]:
    """Where the rating model and the price differ most on who wins (context: the model has not beaten the price)."""
    w = e[(e.kind == "winner") & (e.side == "yes") & e.p_model.notna()].assign(gap=lambda x: x.p_model - x.mid)
    w = w.reindex(w.gap.abs().sort_values(ascending=False).index).head(n)
    return [dict(title=r.title, sport=r.sport, market=round(float(r.mid), 3), model=round(float(r.p_model), 3), gap=round(float(r.gap), 3)) for r in w.itertuples()]


def report(o: dict) -> str:
    """The chief investment officer's written decision."""
    m = lambda v: f"${v:,.2f}"
    st, L = o["state"], []
    L.append(f"\nQUANTSPORT DAILY INVESTMENT REPORT   run {o['run_id']}   mode: {o['mode'].upper()}   model {o['model_version']}")
    L.append(f"Fund: bankroll {m(st['bankroll'])}, cash {m(st['cash'])}, at risk {m(st['exposure'])}, realized {st['realized']:+.2f}, "
             f"drawdown {st['drawdown']:.1%}, risk scale {o['state_before']['scale']:g}.  {o['guard']}")
    if o["state_before"]["stops"]:
        L.append("TRADING STOPPED: " + "; ".join(o["state_before"]["stops"]))
    L.append(f"\n1) MARKETS SCANNED: {o['scanned']:,} buyable sides in {o['games']} games ({o['identified']} matched to the rating models).")
    for r in o["summary"]:
        L.append(f"     {r['sport']:5s} {r['kind']:10s} {r['sides']:6,d} sides, {r['games']:3d} games, best expected return at our estimate {r['best_ret']:+.1%}")
    L.append("\n2) WHERE OUR PROBABILITIES COME FROM")
    seen = set()
    for kx, v in o["how"].items():
        if v not in seen:
            seen.add(v)
            L.append(f"     {kx}: {v}")
    acc = [d for d in o["decisions"] if d["decision"] == "ACCEPT"]
    wait = [d for d in o["decisions"] if d["decision"] == "WAIT"]
    veto = [d for d in o["decisions"] if d["decision"] == "VETO"]
    L.append(f"\n3) TIER 1, CORE SINGLES: {len(acc)} accepted, {len(wait)} to wait on, {len(veto)} vetoed, out of {len(o['decisions'])} sent through the full chain.")
    if not acc:
        L.append("     NO BET on the whole slate. Nothing has a positive expected value at the cautious end of our estimate after the fee.")
    if o.get("ladder_dropped"):
        L.append(f"     ({o['ladder_dropped']} other lines for the same players were set aside: a player's ladder of lines is one position.)")
    for d in acc + wait + veto:
        L.append(f"     {d['decision']:6s} {d['side'].upper()} {d['title'][:52]:52s} ours {d['p_est']:.1%} vs ask {d['ask']:.1%}  EV {d['ev']:+.3f}/contract"
                 + (f"  stake {m(d['stake'])}" if d["decision"] == "ACCEPT" else f"  {d['reason'][:70]}"))
    best = sorted(o["decisions"], key=lambda d: -d["ret"])[:3]
    L.append("     Closest to a bet (still NO BET):")
    for d in best:
        L.append(f"       {d['side'].upper()} {d['title'][:50]:50s} ask {d['ask'] * 100:.0f}c, ours {d['p_est']:.1%} +/- {d['p_se']:.1%}, expected return {d['ret']:+.1%}")
    L.append("\n   Where the rating model disagrees most with the price (context only: the model has not beaten the price):")
    for g in o["disagreements"][:5]:
        L.append(f"       {g['title'][:40]:40s} market {g['market']:.0%}, model {g['model']:.0%} ({g['gap']:+.0%})")
    names = {"small": "TIER 2, SMALL MULTIPLIERS", "diversified": "TIER 3, DIVERSIFIED COMBOS", "degen": "TIER 4, DEGEN"}
    for i, tier in enumerate(("small", "diversified", "degen"), 4):
        cs = [c for c in o["cards"] if c["tier"] == tier]
        keep = o["cfg"]["min_keep"][tier]
        L.append(f"\n{i}) {names[tier]}: {len(cs)} card(s), {m(o['cfg']['card_stake'][tier])} each, weekly allowance {o['cfg']['weekly_spend'][tier]:.0%} of bankroll. "
                 f"Buy only if the quote pays at least {keep:.0%} of the fair payout."
                 + ("  This tier is entertainment money: negative expected value by design, small fixed budget." if tier == "degen" else ""))
        if tier in o["tier_notes"]:
            L.append(f"     {o['tier_notes'][tier]}")
        for c in cs:
            L.append(f"     {c['card_id']}  {c['n_legs']} legs in {len(c['games'])} games ({'/'.join(c['sports'])})  chance {c['p_joint']:.2%} (1 in {1 / c['p_joint']:,.0f})  "
                     f"fair {c['fair_mult']:,.1f}x  WALK-AWAY {c['need_mult']:,.1f}x ({m(c['stake'])} must pay {m(c['max_payout'])})  EV there {c['ev_at_need']:+.0%}  "
                     f"lose-all {c['p_lose']:.1%}  [{c['status']}{': ' + c['vetoes'][0] if c['vetoes'] else ''}]")
    if o["sim"]:
        s = o["sim"]
        L.append(f"\n7) PORTFOLIO if every pending card were bought at its walk-away payout: expected {s['expected']:+.2f}, chance of a losing night {s['p_loss']:.0%}, "
                 f"95% worst case {m(s['var95'])} lost, average of the worst 5% {m(s['cvar95'])}, best case {s['best']:+,.0f}.")
    L.append("\nReal combo quotes need the owner's API key. Measured so far, combo buyers get roughly 72-86c of fair value per dollar, so cards that need")
    L.append("100% of fair will rarely be offered. That is the system declining negative expected value, not a fault.")
    if o["errors"]:
        L.append("Scanner errors: " + "; ".join(o["errors"][:3]))
    return "\n".join(L)
