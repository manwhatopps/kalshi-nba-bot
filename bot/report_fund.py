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
        print(f"   [{e.grade}] {e.edge}")
        print(f"       where it was found : {e.bets} bets, {e.losses} lost, won {e.won:.1%} (break-even {e.cost:.1%}) -> {e.edge_pct:+.1%} per $1  ({e.note})")
        if e.by_season:
            print(f"                            {e.by_season}")
        if e.ind_bets:
            print(f"       independent games  : {e.ind_bets} bets ({e.arch_bets} from last season's archive, {e.new_bets} since discovery), "
                  f"{e.ind_losses} lost, won {e.ind_won:.1%} (break-even {e.ind_cost:.1%}) -> {e.ind_edge:+.1%} per $1")
    print("   A = holds on games it was not found on.  B = looks good, too few independent games to judge.  "
          "C = fails or is indistinguishable from zero: no money.")

    e = primary_edge(reg)
    pre_only = bool(cfg.get("pre_game_only", True))
    if pre_only:
        e = None          # the graded edges above are in-game rules; with pre-game betting only they are not used
    k = data_client()
    from .favorites import live_legs
    try:
        tonight = live_legs(k, "NBA", hours_ahead=args.hours, max_spread=0.2)
        games = tonight[tonight.series == "KXNBAGAME"].game.nunique() if len(tonight) else 0
    except Exception:
        tonight, games = pd.DataFrame(), 0

    print("\n2) CORE SLEEVE")
    if e is None and pre_only:
        frac, exp_tr = 0.0, 0.0
        print("   PRE-GAME ONLY (your setting): every bet is placed before the game, so the in-game rules above are not used.")
        try:
            from .walkforward import pre_active
            act = pre_active()
        except Exception as ex:
            act = pd.DataFrame()
            print(f"   (pre-game rule check unavailable: {ex})")
        if act.empty:
            print("   No pre-game rule clears the bar (positive by 3 std errors on earlier nights, and over its last 20 nights),")
            print("   so the core sleeve stays in CASH tonight. `python -m bot walk --pre` shows the test.")
        else:
            for a in act.itertuples():
                print(f"   ACTIVE pre-game rule: {a.rule}  {a.bets} bets, {a.ret:+.1%} per $1, last 20 nights {a.last_20_nights:+.1%}; "
                      f"stake {_money(min(a.frac * B, cfg['max_stake_per_bet']))} per bet (`python -m bot slip` lists tonight's)")
    elif e is None:
        print("   No edge is graded A or B, so the core sleeve stays in CASH tonight. That is the system working:")
        print("   the rule that looked best this season lost money on last season's 1,200+ games.")
        frac, exp_tr = 0.0, 0.0
    else:
        frac = fund.stake_fraction(e)
        exp_tr = e.game_share * games if e.id != "dd_under" else e.per_night * games / 7.0
        per = min(frac * B, core / max(exp_tr, 1), cfg["max_stake_per_bet"]) if games else min(frac * B, cfg["max_stake_per_bet"])
        win_ret = 1 / e.cost - 1
        print(f"   Edge used: {e.edge}  [{e.grade}]")
        print(f"   Kelly on the measured win rate: {e.kelly_full:.0%} of bankroll ({_money(e.kelly_full * B)}).  "
              f"Kelly on the low end of its error range: {e.kelly_safe:.0%} ({_money(e.kelly_safe * B)}).")
        print(f"   NBA games tonight: {games}.  Expected bets: about {exp_tr:.0f}.")
        if e.id == "dd_under":
            print("   LIQUIDITY: thin markets. A typical one saw about $5 of unders bought at these prices this season; "
                  "the live watcher records the size actually on offer.")
        print(f"   PLAN: {_money(per)} on each bet (capped at {_money(cfg['max_stake_per_bet'])} for liquidity), "
              f"never more than {_money(core)} at risk at once.")
        print(f"         Each win makes {_money(per * win_ret)}; each loss costs {_money(per)}.  "
              f"Expected profit tonight: {_money(exp_tr * per * e.edge_pct)} if the edge is real, $0 if it is not.")
        one_loss = 1 - e.won
        print(f"   The whole {_money(core)} on ONE bet instead: expected profit {_money(core * e.edge_pct)}, but a "
              f"{one_loss:.1%} chance of losing all {_money(core)} on that one bet. Spread over ~{max(exp_tr, 1):.0f} bets, "
              f"losing that much needs nearly every one of them to fail.")

    try:
        if pre_only:
            raise StopIteration
        from .walkforward import active_now
        act = active_now()
        act = act[act.passes_lifetime]
        print("\n   WHAT THE WALK-FORWARD LOGIC WOULD BET TONIGHT (record to date; must be positive 3 std errors down AND over its last 20 nights):")
        for a in act.itertuples():
            flag = "ACTIVE" if a.passes_recent else "paused (recent form negative)"
            print(f"     {flag:30s} {a.rule:38s} {a.bets:5d} bets, {a.ret:+.1%} lifetime, {a.last_20_nights:+.1%} last 20 nights, "
                  f"{a.this_season:+.1%} this season")
    except StopIteration:
        pass
    except Exception as ex:
        print(f"   (walk-forward status unavailable: {ex})")

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
