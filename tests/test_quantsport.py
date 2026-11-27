"""QuantSport: probability, payoff, fees, Kelly, joint probability, correlation, exposure, diversification, bankroll limits,
drawdown controls, data freshness, settlement, approval enforcement, and no-lookahead in the rating model."""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantsport import audit, chain, config, execution, mathx as M, risk, store, tiers
from quantsport.models import ratings as R


def cfg():
    c = config.load()
    c["bankroll"] = 100.0
    return c


def test_payoff_fee_ev_break_even():
    assert abs(M.fee(0.50) - 0.0175) < 1e-12 and M.fee(0.0) == 0 and M.fee(1.0) == 0
    assert M.order_fee(0.56, 8) == math.ceil(0.07 * 8 * 0.56 * 0.44 * 100) / 100           # rounded up to the cent, once per order
    f = M.fee(0.57)
    assert abs(M.ev(0.68, 0.57, f) - (0.68 * 0.43 - 0.32 * 0.57 - f)) < 1e-12              # p(1-c) - (1-p)c - fees
    assert abs(M.ev(M.break_even(0.57, f), 0.57, f)) < 1e-12                                 # worth nothing at break-even
    assert M.ev(0.57, 0.57, f) < 0                                                           # the market's own price loses the fee
    assert abs(M.expected_return(0.68, 0.57, f) - M.ev(0.68, 0.57, f) / (0.57 + f)) < 1e-12
    n, cost = M.contracts_for(5.0, 0.56)
    assert n == 8 and cost <= 5.0 and M.contracts_for(5.0, 0.56)[0] * 0.56 + M.order_fee(0.56, 9) > 0


def test_kelly_binary_contract():
    c = 0.40
    b = (1 - c) / c
    assert abs(M.kelly(0.55, c) - (b * 0.55 - 0.45) / b) < 1e-12                             # f* = (bp - q) / b
    assert abs(M.kelly(0.55, c) - (0.55 - c) / (1 - c)) < 1e-12                              # same thing for a $1 contract
    assert M.kelly(0.40, 0.40) == 0 and M.kelly(0.30, 0.40) == 0                             # no edge, no bet
    t = M.kelly_table(0.55, c, cap=0.05)
    assert t["half"] == t["full"] / 2 and t["quarter"] == t["full"] / 4 and t["capped"] == 0.05
    assert M.kelly_sized(0.55, c, fraction=0.25, cap=0.05) == min(0.25 * M.kelly(0.55, c), 0.05)
    assert M.kelly_sized(0.55, c, p_se=0.06, z=3.0) == 0                                     # edge inside its own uncertainty: zero
    assert M.kelly(0.55, c, fees=0.02) < M.kelly(0.55, c)                                    # fees shrink the bet


def test_joint_probability_and_correlation():
    assert abs(M.joint_independent([0.6, 0.5, 0.5]) - 0.15) < 1e-12
    assert abs(M.joint_copula([0.6, 0.5], np.eye(2)) - 0.30) < 1e-12
    pos = M.joint_copula([0.6, 0.6], [[1, 0.5], [0.5, 1]])
    neg = M.joint_copula([0.6, 0.6], [[1, -0.5], [-0.5, 1]])
    assert neg < 0.36 < pos                                                                  # correlated legs do not multiply
    rho = M.latent_rho(0.6, 0.6, 0.25)                                                       # yes/no correlation 0.25 -> hidden-variable correlation
    assert abs(M.joint_copula([0.6, 0.6], [[1, rho], [rho, 1]]) - (0.36 + 0.25 * 0.24)) < 0.01
    assert abs(M.fair_multiple(0.01) - 100) < 1e-9 and abs(M.combo_ev(0.01, 90) - (-0.10)) < 1e-12
    assert M.devig([0.55, 0.50]) == pytest.approx([0.55 / 1.05, 0.50 / 1.05])


def test_drawdown_controls_and_var():
    steps = [(0.10, 0.5), (0.20, 0.25), (0.30, 0.0)]
    assert [M.drawdown_scale(d, steps) for d in (0.0, 0.05, 0.10, 0.25, 0.30, 0.5)] == [1.0, 1.0, 0.5, 0.25, 0.0, 0.0]
    v, cv = M.var_cvar(np.array([-10.0] * 10 + [1.0] * 90))
    assert v == 10.0 and cv == 10.0


def _pos(con, **kw):
    row = dict(opened=store.now(), mode="paper", kind="single", ref="X:yes", tier="core", sport="NBA", label="x", stake=5.0, pays=9.0, tickers=["X:yes"],
               games=["NBA:G1"], teams=["NBA:BOS"], players=[], expires=None)
    row.update(kw)
    return store.insert(con, "positions", row)


def test_exposure_registry_and_bankroll_limits():
    c, con = cfg(), store.connect(":memory:")
    _pos(con, stake=8.0)
    st, reg = risk.state(con, c), risk.Registry(con)
    assert st["exposure"] == 8.0 and st["cash"] == 92.0 and reg.by["event"]["NBA:G1"] == 8.0 and reg.by["team"]["NBA:BOS"] == 8.0
    ok = dict(stake=1.0, tier="core", kind="single", games=["NBA:G2"], teams=["NBA:LAL"], players=[], sport="NBA", ev_low=0.02, spread=0.02,
              depth_dollars=50, quote_age_s=5, rules="rule text")
    assert risk.audit(ok, reg, st, c) == []
    assert any("event limit" in v for v in risk.audit(dict(ok, stake=3.0, games=["NBA:G1"]), reg, st, c))        # 8 + 3 > 10% of bankroll
    assert any("team limit" in v for v in risk.audit(dict(ok, stake=3.0, teams=["NBA:BOS"]), reg, st, c))
    assert any("per-position cap" in v for v in risk.audit(dict(ok, stake=6.0), reg, st, c))
    assert any("not positive" in v for v in risk.audit(dict(ok, ev_low=-0.01), reg, st, c))                      # never a negative-EV single
    assert any("exceeds available cash" in v for v in risk.audit(dict(ok, stake=95.0), reg, st, c))
    assert any("rule text missing" in v for v in risk.audit(dict(ok, rules=""), reg, st, c))


def test_data_freshness_spread_and_depth():
    c, con = cfg(), store.connect(":memory:")
    st, reg = risk.state(con, c), risk.Registry(con)
    ok = dict(stake=1.0, tier="core", kind="single", games=["NBA:G2"], teams=[], players=[], sport="NBA", ev_low=0.02, spread=0.02, depth_dollars=50,
              quote_age_s=5, rules="r")
    assert any("old" in v for v in risk.audit(dict(ok, quote_age_s=c["quote_max_age_s"] + 1), reg, st, c))        # stale quotes cannot back an order
    assert any("spread" in v for v in risk.audit(dict(ok, spread=0.10), reg, st, c))
    assert any("offered" in v for v in risk.audit(dict(ok, depth_dollars=0.2), reg, st, c))


def test_loss_limits_stop_trading():
    c, con = cfg(), store.connect(":memory:")
    pid = _pos(con, stake=11.0)
    con.execute("UPDATE positions SET status='lost', closed=?, pnl=-11 WHERE id=?", (store.now(), pid))
    con.commit()
    st = risk.state(con, c)
    assert st["scale"] == 0.0 and any("daily loss" in s for s in st["stops"])                                   # -11 on 100 breaches the 10% day stop
    ok = dict(stake=1.0, tier="core", kind="single", games=["NBA:G2"], teams=[], players=[], sport="NBA", ev_low=0.02, spread=0.02, depth_dollars=50,
              quote_age_s=5, rules="r")
    assert any("trading stopped" in v for v in risk.audit(ok, risk.Registry(con), st, c))


def test_card_diversification_rules():
    c, con = cfg(), store.connect(":memory:")
    st, reg = risk.state(con, c), risk.Registry(con)
    leg = lambda g, p="": dict(game_key=g, player=p)
    card = dict(stake=1.0, tier="degen", kind="combo", games=["NBA:G1", "NFL:G2"], teams=[], players=[], sports=["NBA", "NFL"], legs=[leg("NBA:G1"), leg("NFL:G2")])
    assert risk.audit(card, reg, st, c) == []
    same = dict(card, games=["NBA:G1"], legs=[leg("NBA:G1"), leg("NBA:G1")])
    assert any("same-game override" in v for v in risk.audit(same, reg, st, c))                                  # mandatory by default
    assert not any("same-game" in v for v in risk.audit(dict(same, override_same_game=2), reg, st, c))            # only with the explicit override
    reg.add(dict(stake=1.0, tier="degen", games=["NBA:G1", "NFL:G2"], kind="combo", ref="c1", new=True))
    other = dict(card, games=["NBA:G1", "NFL:G9"], legs=[leg("NBA:G1"), leg("NFL:G9")])
    assert any("already on another open card" in v for v in risk.audit(other, reg, st, c))                       # cross-card exposure
    twice = dict(card, games=["NBA:G5", "NFL:G6"], legs=[leg("NBA:G5", "NBA:P1"), leg("NFL:G6", "NBA:P1")])
    assert any("same player" in v for v in risk.audit(twice, reg, st, c))
    for _ in range(10):                                                                                          # the degen pot is 5% of bankroll
        reg.add(dict(stake=1.0, tier="degen", games=[], kind="combo", ref=f"x{_}", new=True))
    assert any("pot cannot cover" in v or "weekly allowance" in v for v in risk.audit(dict(card, games=["NBA:G7", "NFL:G8"], legs=[leg("NBA:G7"), leg("NFL:G8")]), reg, st, c))


def _est(n_games=30, per_game=3):
    rows = []
    for g in range(n_games):
        for j in range(per_game):
            rows.append(dict(ticker=f"KXNFLSPREAD-G{g}-T{j}", event_ticker=f"KXNFLSPREAD-G{g}", side="yes", title=f"g{g} leg{j}", sport="NFL", kind="spread",
                             game_key=f"NFL:G{g}", player="", team_code=f"T{g}", ask=0.62 + 0.01 * j, bid=0.60 + 0.01 * j, mid=0.61 + 0.01 * j, spread=0.02,
                             p_est=0.61 + 0.01 * j, p_model=float("nan"), ret=-0.03, source="price", start=pd.Timestamp("2026-12-01", tz="UTC"), pref=0.0))
    return pd.DataFrame(rows)


def test_tier_builder_one_leg_per_game_and_no_game_on_two_cards():
    c, con = cfg(), store.connect(":memory:")
    c["weekly_spend"] = dict(small=0.03, diversified=0.03, degen=0.02)
    st, reg = risk.state(con, c), risk.Registry(con)
    rules = {f"KXNFLSPREAD-G{g}": (False, 1) for g in range(30)}                # Kalshi's collection: these events, either side, one leg each
    unchecked = tiers.build(_est(), c, st, reg, rules={})
    assert all(x["status"] == "VETO" for t in tiers.TIER_ORDER for x in unchecked["cards"][t])     # no rules loaded -> nothing is waved through
    b = tiers.build(_est(), c, st, reg, rules=rules)
    cards = [x for t in tiers.TIER_ORDER for x in b["cards"][t]]
    assert cards and all(x["status"] == "PENDING QUOTE" for x in cards)
    seen = set()
    for x in cards:
        games = [l["game_key"] for l in x["legs"]]
        assert len(games) == len(set(games))                                   # one leg per game on a card
        assert not (set(games) & seen)                                         # no game on two cards
        seen |= set(games)
        assert abs(x["p_joint"] - np.prod([l["p"] for l in x["legs"]])) < 1e-12 and abs(x["fair_mult"] - 1 / x["p_joint"]) < 1e-9
        assert abs(x["need_mult"] - x["fair_mult"] * c["min_keep"][x["tier"]]) < 1e-9 and x["max_loss"] == x["stake"]
    small = [x for x in cards if x["tier"] == "small"]
    assert all(c["small_range"][0] <= x["fair_mult"] <= c["small_range"][1] and x["n_legs"] == 2 for x in small)
    assert all(x["ev_at_need"] <= 1e-9 for x in cards if x["tier"] == "degen")   # no card is presented as a likely winner
    tiny = tiers.build(_est(n_games=3, per_game=1), c, st, risk.Registry(con), rules=rules)
    assert not tiny["cards"]["degen"] and "NO DEGEN CARD" in tiny["notes"]["degen"]     # never manufacture a card


def test_chain_says_no_bet_at_the_market_price_and_accepts_only_a_proven_edge():
    c, con = cfg(), store.connect(":memory:")
    st, reg = risk.state(con, c), risk.Registry(con)
    base = dict(ticker="KXNBAGAME-G1-BOS", side="yes", sport="NBA", kind="winner", game_key="NBA:G1", title="Boston wins", rules="If Boston wins...",
                ask=0.60, bid=0.58, mid=0.59, spread=0.02, ask_size=500, team_code="BOS", player="", scanned=datetime.now(timezone.utc),
                p_model=0.70, p_est=0.59, p_se=0.03, source="price")
    d = chain.decide(base, c, st, reg)
    assert d["decision"] == "NO BET" and d["stake"] == 0 and len(d["thesis"]["steps"]) == 16
    assert any("NOT VERIFIED" in a for a in d["thesis"]["availability"])                          # never pretends to have checked injuries
    good = chain.decide(dict(base, p_est=0.80, p_se=0.02, source="blend"), c, st, reg)
    assert good["decision"] == "ACCEPT" and 0 < good["stake"] <= c["kelly_cap"] * 100 + 1e-9 and good["ev_low"] > 0
    wait = chain.decide(dict(base, p_est=0.66, p_se=0.03, source="blend"), c, st, reg)
    assert wait["decision"] == "WAIT" and wait["wait_price"] < base["ask"]
    stale = chain.decide(dict(base, p_est=0.80, p_se=0.02, scanned=datetime.now(timezone.utc) - timedelta(seconds=600)), c, st, reg)
    assert stale["decision"] == "VETO" and any("old" in v for v in stale["vetoes"])


def test_settlement_logic():
    g = audit.grade_position
    assert g("single", ["A:yes"], {"A": "yes"}) == ("won", 1) and g("single", ["A:no"], {"A": "yes"}) == ("lost", 0)
    assert g("combo", ["A:yes", "B:no"], {"A": "yes"}) == ("open", 1)                             # not won until every leg is in
    assert g("combo", ["A:yes", "B:no"], {"B": "yes"}) == ("lost", 0)                             # lost as soon as one leg loses
    assert g("combo", ["A:yes", "B:no"], {"A": "yes", "B": "no"}) == ("won", 2)


def test_approval_enforcement(tmp_path, monkeypatch):
    monkeypatch.setenv("QS_APPROVAL_KEY", str(tmp_path / "approval.key"))
    con = store.connect(":memory:")
    order = dict(ticker="KXTEST-1", side="no", limit_price=0.56, quantity=8, max_cost=4.70)
    with pytest.raises(PermissionError):
        execution.issue(con, order)                                                               # no key, no approval
    with pytest.raises(PermissionError):
        execution.approve(con, order)                                                             # not a person at a terminal
    execution._key(create=True)
    tok = execution.issue(con, order)
    assert execution.verify(con, order, tok) == (True, "ok")
    for changed in (dict(order, quantity=9), dict(order, limit_price=0.60), dict(order, max_cost=9.0), dict(order, side="yes"), dict(order, ticker="KXTEST-2")):
        assert execution.verify(con, changed, tok)[0] is False                                    # bound to the exact order
    assert execution.verify(con, order, "nope")[0] is False

    class K:
        authed = True
        sent = []

        def request(self, *a, **kw):
            K.sent.append(kw["json"])
            return {"ok": True}
    live = dict(mode="live_review", quote_max_age_s=120)
    with pytest.raises(PermissionError):
        execution.place_approved(K(), con, order, tok, dict(live, mode="paper"), dict(ask=0.55, age_s=5))       # wrong mode
    with pytest.raises(PermissionError):
        execution.place_approved(K(), con, order, tok, live, dict(ask=0.60, age_s=5))             # price above the approved limit (and the token is now spent)
    assert execution.verify(con, order, tok) == (False, "approval already used") and K.sent == []
    tok2 = execution.issue(con, order)
    with pytest.raises(PermissionError):
        execution.place_approved(K(), con, order, tok2, live, dict(ask=0.55, age_s=999))          # stale quote
    tok3 = execution.issue(con, order)
    assert execution.place_approved(K(), con, order, tok3, live, dict(ask=0.55, age_s=5)) == {"ok": True}
    assert K.sent == [dict(ticker="KXTEST-1", side="ask", count="8.00", price="0.4400", time_in_force="immediate_or_cancel", client_order_id=tok3)]
    with pytest.raises(PermissionError):
        execution.place_approved(K(), con, order, tok3, live, dict(ask=0.55, age_s=5))            # no reuse
    tok4 = execution.issue(con, order, ttl_min=-1)
    assert execution.verify(con, order, tok4) == (False, "approval expired")
    assert execution.tickets([dict(decision="NO BET", contracts=0)]) == []


def test_rating_model_never_looks_ahead_and_prices_markets_consistently():
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(8)]
    rows = [dict(date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i // 4), season="2026", home=teams[a], away=teams[b],
                 home_pts=float(100 + rng.integers(-15, 16)), away_pts=float(100 + rng.integers(-15, 16)), neutral=False)
            for i, (a, b) in enumerate((int(x), int(y)) for x, y in rng.integers(0, 8, (300, 2)) if x != y)]
    g = pd.DataFrame(rows)
    a, _ = R.walk("NBA", g)
    h = g.copy()
    h.loc[h.index >= 150, ["home_pts", "away_pts"]] = 999.0                    # change the future only
    b, _ = R.walk("NBA", h)
    pd.testing.assert_series_equal(a.mu.iloc[:151], b.mu.iloc[:151])           # predictions up to there cannot move
    P = R.Pred(mu=4.0, sigma=12.0, total=220.0, sigma_t=18.0, games_home=20, games_away=20)
    assert abs(P.p_win(True) + P.p_win(False) - 1) < 1e-12 and P.p_win(True) > 0.5
    assert P.p_margin_over(4.0, True) == pytest.approx(0.5) and P.p_margin_over(10.5, True) < P.p_margin_over(3.5, True)
    assert P.p_total_over(220.0) == pytest.approx(0.5) and P.p_team_over(112.0, True) == pytest.approx(0.5)


def test_portfolio_simulation_and_kelly_portfolio():
    s = risk.simulate_portfolio([dict(stake=5, pays=10, legs=[(0.5, "g1", "")])], sims=40_000)
    assert abs(s["expected"]) < 0.15 and abs(s["p_loss"] - 0.5) < 0.02 and s["worst"] == -5 and s["best"] == 5
    two = risk.simulate_portfolio([dict(stake=1, pays=4, legs=[(0.5, "g1", "p"), (0.5, "g1", "p")])], sims=40_000)     # same player, tied legs
    assert two["p_loss"] < 0.75                                                # hits more often than 25%: not independent
    f = risk.kelly_portfolio(np.array([0.62, 0.55, 0.70, 0.50]), np.array([0.55, 0.56, 0.60, 0.50]), ["a", "b", "a", "c"], cap=0.05, budget=0.5, group_cap=0.08)
    assert f[1] == 0 and f[3] == 0 and f[0] > 0 and f[2] > 0 and f[0] + f[2] <= 0.08 + 1e-6 and (f <= 0.05 + 1e-9).all()


def test_injury_flags_use_only_what_was_known_before_the_week():
    from bot import nfl_form as F
    rows = []
    for wk in range(1, 6):
        d = pd.Timestamp("2026-09-06") + pd.Timedelta(days=7 * (wk - 1))
        for pid, pos, tgt, car, att in (("QB1", "QB", 0, 2, 35), ("RB1", "RB", 3, 18, 0), ("RB2", "RB", 1, 5, 0), ("WR1", "WR", 9, 0, 0)):
            rows.append(dict(player_id=pid, season=2026, week=wk, game_date=d, team="AAA", position=pos, targets=tgt, carries=car, attempts=att))
    L = pd.DataFrame(rows)
    inj = pd.DataFrame([dict(season=2026, week=5, gsis_id="RB1", report_status="Out", report_primary_injury="Knee", team="AAA"),
                        dict(season=2026, week=5, gsis_id="WR1", report_status="Questionable", report_primary_injury="Ankle", team="AAA"),
                        dict(season=2026, week=6, gsis_id="QB1", report_status="Out", report_primary_injury="Shoulder", team="AAA"),
                        dict(season=2026, week=7, gsis_id="RB2", report_status=None, report_primary_injury="Hamstring", team="AAA")])
    f = F.week_flags(L, inj, 2026, 5).set_index("player_id")
    assert f.loc["RB2", "lead_out"] == 1.0 and f.loc["RB1", "lead_out"] == 0.0        # the backup benefits, not the injured starter himself
    assert f.loc["WR1", "status"] == "Questionable" and f.loc["WR1", "lead_out"] == 0.0
    assert f.qb_out.sum() == 0                                                          # next week's report is not visible in week 5
    g = F.week_flags(L, inj, 2026, 6).set_index("player_id")
    assert g.loc["WR1", "qb_out"] == 1.0 and g.loc["QB1", "qb_out"] == 0.0
    assert g.team_posted.all() and not F.week_flags(L, inj, 2026, 7).team_posted.any()   # practice report only: statuses not posted yet


def test_band_gaps_shrink_toward_the_overall_gap_and_never_claim_more_certainty():
    from bot import legs
    rng = np.random.default_rng(0)
    cost = rng.uniform(0.2, 0.85, 3000)
    sub = pd.DataFrame(dict(no_cost=cost, win=(rng.random(3000) < cost + 0.05).astype(float)))
    b = legs.band_gaps(sub, 0.05, 0.02)
    assert [x[:2] for x in b] == [list(x) for x in legs.BANDS] and sum(x[3] for x in b) == 3000
    assert all(x[4] >= 0.02 for x in b)                                                 # a band is never surer than the whole category
    raw = float((sub[sub.no_cost < 0.45].win - sub[sub.no_cost < 0.45].no_cost).mean())
    assert min(raw, 0.05) - 1e-9 <= b[0][2] <= max(raw, 0.05) + 1e-9                    # between the band's own number and the overall one
    assert legs.band_gaps(sub.iloc[:0], 0.05, 0.02)[0][2:] == [0.05, 0, 0.02]


def test_a_players_ladder_is_one_position():
    from quantsport import pipeline
    e = pd.DataFrame([dict(sport="NFL", player="S Barkley", series="KXNFLRSHYDS", side="no", ticker=f"T{i}", ret_low=r) for i, r in enumerate((-0.05, -0.01, -0.03))]
                     + [dict(sport="NFL", player="S Barkley", series="KXNFLREC", side="no", ticker="R1", ret_low=-0.02),
                        dict(sport="NFL", player="", series="KXNFLGAME", side="yes", ticker="G1", ret_low=-0.04),
                        dict(sport="NFL", player="", series="KXNFLGAME", side="yes", ticker="G2", ret_low=-0.06)])
    out = pipeline.one_per_ladder(e)
    assert sorted(out.ticker) == ["G1", "G2", "R1", "T1"]


def test_watch_list_record_counts_a_ladder_once_and_reports_closing_value(tmp_path):
    con = store.connect(tmp_path / "t.db")
    base = dict(run_id="r", ts="2026-11-26T22:00:00+00:00", sport="NFL", kind="prop", side="no", decision="WAIT", fee=0.01, source="category: x")
    for line, ask, res in ((50, 0.30, "lost"), (60, 0.48, "won"), (70, 0.60, "won")):
        store.insert(con, "decisions", base | dict(ticker=f"KXNFLRSHYDS-26NOV26KCBUF-KCJOTT34-{line}", title="t", ask=ask, p_est=ask + 0.05, result=res, close_price=ask + 0.02))
    store.insert(con, "decisions", base | dict(ticker="KXNFLRSHYDS-26NOV26KCBUF-KCPMAHOMES15-20", title="t", ask=0.55, p_est=0.6, result="lost", close_price=0.53))
    store.insert(con, "decisions", base | dict(ticker="KXNFLRSHYDS-26NOV26KCBUF-BUFJALLEN17-30", title="t", ask=0.5, p_est=0.5, result="won", decision="NO BET"))
    con.commit()
    sh = audit.shadow_record(con)
    assert sh["n"] == 2 and sh["lines"] == 4 and sh["games"] == 1                    # one position per player; the NO BET is not on the watch list
    assert sh["won"] == 0.5 and abs(sh["needed"] - (0.49 + 0.56) / 2) < 1e-9
    assert abs(sh["ret"] - (1 - 0.49 - 0.56) / (0.49 + 0.56)) < 1e-9 and abs(sh["clv"] - 0.0) < 1e-9


def test_position_monitor_values_a_card_and_hedges_only_the_last_leg():
    from quantsport import monitor
    pos = dict(label="small 2.0x", stake=1.0, pays=2.0)
    won, lost = dict(ticker="A", side="yes", state="won"), dict(ticker="B", side="yes", state="lost")
    p1 = dict(ticker="C", side="yes", state="pending", mid=0.6, opp_ask=0.42)
    p2 = dict(ticker="D", side="no", state="pending", mid=0.5, opp_ask=0.52)
    two = monitor.position_status(pos, [won, p1, p2], 100)
    assert abs(two["alive"] - 0.3) < 1e-9 and abs(two["value"] - 0.6) < 1e-9 and two["hedge"] is None       # nothing to hedge with two legs left
    one = monitor.position_status(pos, [won, p1], 100)
    h = one["hedge"]
    assert h["opposite"] == "no" and h["opposite_cost"] > 0.42                                         # the fee is in the hedge's cost
    lock = h["plans"]["lock"]
    assert abs(lock["if_hits"] - lock["if_misses"]) < 1e-9 and lock["ev_given_up"] > 0                 # a lock pays the same either way and costs EV
    assert h["plans"]["break-even"]["if_misses"] == pytest.approx(0.0)
    dead = monitor.position_status(pos, [won, lost, p1], 100)
    assert dead["dead"] and dead["value"] == 0.0 and dead["hedge"] is None
    done = monitor.position_status(pos, [won, dict(won, ticker="Z")], 100)
    assert done["value"] == 2.0 and done["legs_left"] == 0
    assert monitor.position_status(pos, [won, dict(p1, mid=None, opp_ask=None)], 100)["value"] is None    # no price: say so, do not guess


def test_resting_order_counts_a_fill_only_when_the_price_trades_through_the_bid():
    from bot import resting
    o = pd.DataFrame(dict(ticker=list("abcd"), ask=[0.50, 0.50, 0.50, 0.50], low=[0.47, 0.48, 0.49, np.nan], win=[1.0, 0.0, 1.0, 1.0],
                          game=["g1", "g1", "g2", "g3"], side="no", season="this season", league="NFL"))
    s = resting.stat(o, 2)                                    # bid 48c: only the 47c print is a certain fill; 48c is a touch
    assert s["orders"] == 4 and s["filled"] == 1 and s["fill"] == 0.25 and s["touch"] == 0.5
    assert s["ret_touch"] == pytest.approx((1 - 0.48 - 0.48) / 0.96)
    assert s["per_order"] == pytest.approx((1 - 0.48) / 4)
    cost = 0.5 + 0.07 * 0.25
    assert s["taker"] == pytest.approx((3 - 4 * cost) / (4 * cost))


def test_nba_injury_tags_follow_the_reports_own_wording():
    from bot import nba_injuries as NI
    assert NI.tag_of("Out", "GTD", "Nurkic (foot) is questionable for Friday's game against the Nuggets.") == "questionable"
    assert NI.tag_of("Out", "OUT", "Landale (hip) is out for Friday's game against the Celtics.") == "out"
    assert NI.tag_of("Day-To-Day", "GTD", "Anunoby (calf) is doubtful for Friday.") == "doubtful"
    assert NI.tag_of("Day-To-Day", "GTD", "Embiid is dealing with an illness.") == "day-to-day"
    raw = {"injuries": [{"displayName": "Utah Jazz", "injuries": [
        {"athlete": {"displayName": "Jusuf Nurkić"}, "status": "Out", "date": "2026-11-27T00:08Z", "shortComment": "Nurkic (foot) is questionable for Friday.",
         "details": {"fantasyStatus": {"abbreviation": "GTD"}, "type": "Foot", "detail": "Strain", "side": "Right"}}]}]}
    d = NI.parse(raw)
    assert d.key[0] == "jusuf nurkic" and d.tag[0] == "questionable" and d.injury[0] == "right foot strain"      # accents never break the match
    line = NI.status_line(d.iloc[0])
    assert line.startswith("injury report: listed QUESTIONABLE (right foot strain)")
    assert NI.blocks([line]) == "questionable" and NI.blocks(["injury report: listed PROBABLE (thumb), updated x"]) == ""
    assert NI.blocks(["injury report: not listed (expected to play)", "injury report: teammates out who played in the last 3 weeks: A (30 min a game, out)"]) == ""
    assert NI.blocks(["injury report: listed OUT (knee)"]) == "out"                                                 # the NFL line reads the same way


def test_a_listed_player_is_never_accepted_and_never_on_a_card():
    con = store.connect(":memory:")
    c = cfg()
    st = risk.state(con, c)
    base = dict(ticker="KXNBAPTS-26NOV27UTADEN-UTAJNURKIC30-15", event_ticker="KXNBAPTS-26NOV27UTADEN", side="no", title="Jusuf Nurkić: 15+ points", sport="NBA", kind="prop",
                game_key="NBA:26NOV27UTADEN", team_code="UTA", player="Jusuf Nurkić", ask=0.50, bid=0.48, mid=0.49, spread=0.02, ask_size=500, p_est=0.75, p_se=0.01,
                p_model=float("nan"), source="category: x", rules="settles on the box score", scanned=datetime.now(timezone.utc), start=datetime.now(timezone.utc) + timedelta(hours=3))
    ok = chain.decide(base | dict(why=["injury report: not listed (expected to play)"]), c, st, risk.Registry(con))
    q = chain.decide(base | dict(why=["injury report: listed QUESTIONABLE (right foot strain), updated Nov 26 7:08pm ET"]), c, st, risk.Registry(con))
    out = chain.decide(base | dict(why=["injury report: listed OUT (knee), updated Nov 26 7:08pm ET"]), c, st, risk.Registry(con))
    assert ok["decision"] == "ACCEPT" and q["decision"] == "WAIT" and out["decision"] == "NO BET"
    assert "questionable" in q["reason"] and "listed out" in out["reason"]
    e = pd.DataFrame([dict(base, why=w, ret=-0.01, pref=0.0, ask=0.6) for w in (["injury report: not listed (expected to play)"], ["injury report: listed QUESTIONABLE (foot)"],
                                                                         ["injury report: listed DOUBTFUL (calf)"])])
    pool = tiers.leg_pool(e, c, {})
    assert len(pool) == 1 and pool.iloc[0].why == ["injury report: not listed (expected to play)"]


def test_a_league_wide_break_is_not_a_players_own_absence(monkeypatch):
    from bot import legs
    days = [pd.Timestamp("2026-11-01") + pd.Timedelta(days=2 * i) for i in range(8)]            # the team plays every other day ...
    monkeypatch.setattr(legs, "_TEAM_DATES", {"AAA": days + [pd.Timestamp("2026-11-30")], "__max__": [pd.Timestamp("2026-11-30")]})
    assert legs.missed_games("AAA", "2026-11-15", 8) == 3            # he last played Nov 7; the team played Nov 9, 11, 13 without him
    assert legs.team_idle("AAA", "2026-11-15") == 2
    assert legs.missed_games("AAA", "2026-11-30", 15) == 0 and legs.team_idle("AAA", "2026-11-30") == 15     # the whole team was off
    d = pd.DataFrame(dict(missed=[3.0, 0.0, 0.0], team_idle=[2.0, 15.0, 15.0], days_rest=[8.0, 15.0, 15.0], log_fresh=[1.0, 1.0, 0.0]))
    own = legs.NBA["first game back after missing 2+ of his team's games"](d)
    brk = legs.NBA["his team's first game after 7+ days without one (season opener, All-Star break)"](d)
    assert list(own) == [True, False, False] and list(brk) == [False, True, False]                # a stale game log never calls a break
