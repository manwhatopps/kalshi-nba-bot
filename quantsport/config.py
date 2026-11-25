"""Every risk number in one place. Defaults below; `data/quantsport.json` overrides any of them."""
from __future__ import annotations

import json

from bot import config as botcfg

PATH = botcfg.DATA_DIR / "quantsport.json"
DEFAULT = dict(
    mode="paper",                    # research | paper | live_review   (live_review only PREPARES orders for approval)
    bankroll=100.0,
    # --- how sure the engine must be -------------------------------------------------------------
    z_edge=3.0,                      # a single bet needs positive EV at the estimate minus this many standard errors (3: the
                                     # edges on file were found by searching many rules, so the bar is the walk-forward one)
    min_ev=0.0,                      # ...and at least this expected return per dollar at that low-end estimate
    kelly_fraction=0.25,             # default conservative fractional Kelly
    kelly_cap=0.05,                  # never more than this share of bankroll in one position
    max_spread=0.06,                 # widest bid/ask a market may have to be considered
    min_depth_dollars=5.0,           # must be able to fill at least this much inside the limit
    quote_max_age_s=120,             # prices older than this cannot back an order
    data_max_age_h=36,               # stats older than this are flagged on every thesis
    # --- tier budgets: share of bankroll that may be at risk in each strategy at once ---------------
    # core: the most that may be open at once. Card tiers: a POT of capital set aside once. Losses come out of the pot, wins go
    # back in, and when a pot cannot cover a card that tier stops. So the most the cards can ever lose is the sum of their pots.
    # (Simulation: any steady card habit paid out of the general bankroll hits the drawdown stop within months, even at fair quotes.)
    budget=dict(core=0.50, small=0.10, diversified=0.05, degen=0.05),
    weekly_spend=dict(small=0.03, diversified=0.02, degen=0.01),     # pace: share of bankroll a card tier may spend per rolling 7 days
    card_stake=dict(small=1.0, diversified=1.0, degen=1.0),          # dollars per card
    small_range=(1.5, 3.0),          # payout multiples for small multipliers
    diversified_targets=(5, 10, 25),
    degen_targets=(100, 250, 500, 1000),
    min_keep=dict(small=1.0, diversified=1.0, degen=0.90),           # offered payout / fair payout a card needs before it is bought
    leg_price=(0.50, 0.85),          # price range for card legs
    same_game_per_card=1,            # the diversification rule; raising it is the explicit override
    reuse_game_across_cards=False,
    card_horizon_h=120,              # how far ahead a card may reach for games
    # --- exposure limits (share of bankroll) -------------------------------------------------------
    limit=dict(event=0.10, team=0.10, player=0.05, sport=0.60, strategy=0.50, correlated=0.15),
    max_daily_loss=0.10, max_weekly_loss=0.20, max_drawdown=0.30,
    drawdown_steps=((0.10, 0.5), (0.20, 0.25), (0.30, 0.0)),         # at this drawdown, scale new risk by this factor
    calibration_guard=dict(min_settled=100, max_brier_excess=0.01),  # if our estimates score worse than the market by this, halve risk
    sports=("NBA", "NFL", "NCAAF"),
)


def load() -> dict:
    c = json.loads(json.dumps(DEFAULT))          # deep copy, tuples become lists
    if PATH.exists():
        user = json.loads(PATH.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(c.get(k), dict):
                c[k].update(v)
            else:
                c[k] = v
    try:                                          # one bankroll for the whole project
        from bot import fund
        c["bankroll"] = float(json.loads(PATH.read_text()).get("bankroll")) if PATH.exists() and "bankroll" in json.loads(PATH.read_text()) else float(fund.load_cfg()["bankroll"])
    except Exception:
        pass
    return c


def save(changes: dict) -> dict:
    cur = json.loads(PATH.read_text()) if PATH.exists() else {}
    cur.update(changes)
    PATH.write_text(json.dumps(cur, indent=1))
    return load()
