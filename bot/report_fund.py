"""`python -m bot fund`: tonight's allocation plan for the whole bankroll, plus what to expect."""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from . import fund, ledger
from .kalshi import data_client

pd.set_option("display.width", 230)


def _money(v):
    return f"${v:,.2f}" if abs(v) < 100 else f"${v:,.0f}"


def primary_edge(reg: pd.DataFrame) -> pd.Series | None:
    """Strongest graded edge: most standard errors between the win rate and break-even."""
    ok = reg[reg.grade.str[0].isin(["A", "B"])]
    if ok.empty:
        return None
    return ok.assign(zs=(ok.won - ok.cost) / ok.se).sort_values(["grade", "zs"], ascending=[True, False]).iloc[0]


def run(args):
    cfg = fund.load_cfg()
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if args.set:
        fund.save_cfg(cfg)
    B = cfg["bankroll"]
    core, degen_total = cfg["core_pct"] * B, sum(s for s, _ in cfg["degen_tickets"])
    print(f"\nFUND  bankroll {_money(B)}   core sleeve {_money(core)} ({cfg['core_pct']:.0%})   "
          f"degen sleeve {_money(degen_total)}   reserve {_money(B - core - degen_total)}")
    print(f"      stop for the night if the fund is down {cfg['stop_loss_pct']:.0%} ({_money(cfg['stop_loss_pct'] * B)}).")

    reg = fund.edge_registry(cfg["z"])
    print("\n1) EDGES ON FILE (one bet per game; re-graded daily as new games arrive)")
    if reg.empty:
        print("   none yet: run `python -m bot trades --download`")
        return
    for e in reg.itertuples():
        new = f"{e.new_bets} new bets, {e.new_won:.1%} won" if e.new_bets else "no new games yet"
        print(f"   [{e.grade}] {e.edge}: {e.bets} bets, {e.losses} lost, won {e.won:.1%} at {e.price * 100:.1f}c "
              f"(break-even {e.cost:.1%}) -> {e.edge_pct:+.1%} per $1. Since discovery: {new}.")
    print("   A = confirmed on games after discovery.  B = positive even at the low end of its error range, "
          "not yet confirmed.  C = not distinguishable from zero: no money.")

    e = primary_edge(reg)
    k = data_client()
    from .favorites import live_legs
    try:
        tonight = live_legs(k, "NBA", hours_ahead=args.hours, max_spread=0.2)
        games = tonight[tonight.series == "KXNBAGAME"].game.nunique() if len(tonight) else 0
    except Exception:
        tonight, games = pd.DataFrame(), 0

    print("\n2) CORE SLEEVE")
    if e is None:
        print("   No edge is graded A or B, so the core sleeve stays in cash tonight.")
        frac, exp_tr = 0.0, 0.0
    else:
        frac = fund.stake_fraction(e)
        exp_tr = e.game_share * games
        per = min(frac * B, core / max(exp_tr, 1)) if games else frac * B
        win_ret = 1 / e.cost - 1
        print(f"   Edge used: {e.edge}  [{e.grade}]")
        print(f"   Kelly on the measured win rate: {e.kelly_full:.0%} of bankroll ({_money(e.kelly_full * B)}).  "
              f"Kelly on the low end of its error range: {e.kelly_safe:.0%} ({_money(e.kelly_safe * B)}).")
        print(f"   NBA games tonight: {games}.  Expected triggers: about {exp_tr:.0f} (the rule fired in {e.game_share:.0%} of games).")
        print(f"   PLAN: {_money(per)} on each trigger, never more than {_money(core)} at risk at once.")
        print(f"         Each win makes {_money(per * win_ret)}; each loss costs {_money(per)}.  "
              f"Expected profit tonight: {_money(exp_tr * per * e.edge_pct)} if the edge is real, $0 if it is not.")
        one_loss = 1 - e.won
        print(f"   The same {_money(core)} on ONE trigger: same expected profit ({_money(core * e.edge_pct)}), but a "
              f"{one_loss:.1%} chance of losing {_money(core)} in one game. Spread over ~{max(exp_tr, 1):.0f} games that "
              f"single-night loss needs every one of them to fail.")

    print("\n3) DEGEN SLEEVE (lottery tickets: fixed cost, never sized up)")
    run_id = uuid.uuid4().hex[:8]
    try:
        from . import trades as T
        from .favorites import build_best
        parts = [live_legs(k, lg, hours_ahead=args.hours, max_spread=0.06).assign(league=lg) for lg in ("NBA", "NFL")]
        now = pd.concat([p for p in parts if len(p)], ignore_index=True)
        calib = json.loads(T.CALIB.read_text()) if T.CALIB.exists() else T.leg_calibration()
    except Exception as ex:
        now = pd.DataFrame()
        print(f"   could not load tonight's legs: {ex}")
    for stake, goal in cfg["degen_tickets"]:
        need = goal / stake
        if now.empty:
            break
        c = build_best(now, calib, "NBA", need / 0.8, 0.35, 0.62, 2, 0.03, groups=("1 moneyline", "4 player props"))
        if c is None or len(c) < 3:
            print(f"   {_money(stake)} -> {_money(goal)}: not enough tight-book legs tonight")
            continue
        mid, p = float(np.prod(c.mid)), float(np.prod(c.p_hat))
        print(f"\n   {_money(stake)} to win {_money(goal)} (needs {need:,.0f}x): {len(c)} legs across {c.game.nunique()} games")
        for r in c.itertuples():
            print(f"      {r.side.upper():3s} {r.cost:.2f}  {r.title[:62]}")
        print(f"      fair payout {1 / mid:,.0f}x (hits about 1 in {1 / p:,.0f}).  Only buy if the quote pays at least "
              f"{need:,.0f}x: that is {need * mid:.0%} of fair value.")
        print(f"      Expected cost of this ticket at that quote: {_money(stake * (1 - need * p))} of the {_money(stake)}.")
        ledger.log_card(run_id, c, f"degen_{goal}", p, 1 / need, stake=stake, note=f"fund degen {stake}->{goal}")
    print("   Reference: the 16-leg combo you shared sold at 685x when its legs were worth 4,671x (15% of fair value).")

    if e is not None:
        print(f"\n4) WHAT TO EXPECT over {args.nights} game nights (4,000 simulated funds each; lottery tickets assumed to return "
              f"{args.degen_keep:.0%} of fair value)")
        be = e.cost
        rows = []
        for sc, p in (("edge as measured", e.won), ("edge half as big", (e.won + be) / 2), ("no edge at all", be)):
            for name, style, dg in (("your way: whole core on one bet + tickets nightly", "one", "nightly"),
                                    ("spread core + tickets nightly", "spread", "nightly"),
                                    ("spread core + tickets weekly", "spread", "weekly"),
                                    ("spread core + tickets from profits only", "spread", "profits"),
                                    ("spread core, no tickets", "spread", "none")):
                s = fund.simulate(cfg, p, e.price, exp_tr or e.per_night, style, args.nights, dg, args.degen_keep,
                                  stake_frac=frac)
                rows.append(dict(if_=sc, plan=name, median_end=_money(s["median"]), bad_1in10=_money(s["p10"]),
                                 good_1in10=_money(s["p90"]), ahead=f"{s['ahead']:.0%}", halved=f"{s['halved']:.0%}",
                                 core_avg=f"{s['core']:+,.0f}", hit_a_big_one=f"{s['big_hit']:.1%}"))
        print(pd.DataFrame(rows).rename(columns={"if_": "if"}).to_string(index=False))
        print("   core_avg = average dollars the core sleeve made.  halved = chance the fund ends below half.  "
              "hit_a_big_one = chance at least one ticket paid $500+.")
