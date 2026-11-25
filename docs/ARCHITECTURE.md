# QuantSport architecture

One Python package (`quantsport/`) on top of the existing research bot (`bot/`). No services, no queue: a daily
batch that reads public Kalshi data, writes one SQLite file and one HTML page. Every stage is a plain function on
one shared table of candidates, so each can be tested alone.

```
scanner.scan            every open NBA / NFL / NCAAF game and player market, both sides, with rule text and quotes
   |
estimate.model_probs    rating-model probability for game markets (models/ratings.py, walk-forward by construction)
estimate.add_estimates  OUR probability + standard error + where it came from (models/stack.py decides: a model is
   |                    used only where it beat Kalshi's price on later data; otherwise our probability is the price)
chain.decide            the 16-step logical chain per candidate -> ACCEPT / WAIT / VETO / NO BET, with a thesis that
   |                    separates facts, model outputs, assumptions, judgments, and what was NOT verified
risk.audit              independent auditor: limits, pots, weekly allowances, stale quotes, card rules. Cannot be
   |                    switched off by the builder; only data/quantsport.json changes a limit
tiers.build             SMALL / DIVERSIFIED / DEGEN cards: one leg per game, no game on two cards, Kalshi combo rules
   |
pipeline.run            records everything (store.py), opens paper positions in paper mode, writes the CIO report
audit.settle / report   grades positions and decisions; calibration of our probability vs the market's, by sport/market
dashboard.write         data/quantsport/dashboard.html
execution               the only code that can send an order; needs live_review mode + API key + a signed one-use approval
```

## Departments (spec section 12) as modules

| role | code |
|---|---|
| NBA quant analyst | `models/ratings.py` (NBA), `bot/scenarios.py`, `bot/matchup.py`, `bot/legs.py` (NBA categories) |
| Football quant analyst | `models/ratings.py` (NFL, NCAAF), `bot/nfl_form.py`, `bot/nfl_scenarios.py`, `bot/legs.py` (NFL categories) |
| Market intelligence | `scanner.py`, `bot/book.py` (order book), `bot/arb.py`, `bot/combo_check.py` |
| Probability and mathematics | `mathx.py`, `models/stack.py`, `estimate.py` |
| Portfolio manager | `risk.state`, `risk.Registry`, `risk.kelly_portfolio`, `config.py` |
| Hedge analyst | `bot/hedge.py` (lock / break-even / Kelly hedge for a live card) |
| Strategy builder | `chain.py` (core), `tiers.py` (cards) |
| Independent risk auditor | `risk.audit` (separate function, separate inputs, hard vetoes) |
| Performance researcher | `audit.py`, `backtest.py`, `bot/walkforward.py` |
| Chief investment officer | `pipeline.report` |

They are modules, not chat agents: the auditor's independence comes from it being code the builder cannot edit at
run time, which is stronger than a second prompt.

## Source of truth

* `data/quantsport.db` (SQLite): `runs`, `decisions` (every candidate sent through the full chain, with its thesis
  and model version), `cards`, `positions` (paper book), `approvals`, `equity`.
* `data/qs_scans/<run>.parquet`: every side scanned in a run with our estimate at the time (for calibration).
* `data/quantsport_models.json`: tuned rating parameters and the price-vs-model test results the estimator obeys.
* `data/quantsport.json`: risk configuration overrides (defaults in `quantsport/config.py`).

## The estimate rule (why almost everything is NO BET)

`models/stack.py` regresses the result on logit(price) and logit(model) on the earlier 60% of dates and scores the
later 40%. A model moves our probability only if the blend beats the price there AND the model's weight is over
2 standard errors. On 2026-11-25 that is true for none of nine sport x market combinations (57,516 settled
markets), and for no player-prop model (FINDINGS 2, 17, 19, 21). So our probability is Kalshi's price, the
expected value of taking it is minus the fee and half the spread, and the chain says NO BET. Graded leg categories
(`bot/legs.py`) can raise an estimate above the price, but a single still has to clear 3 standard errors.

## Risk design in one paragraph

Core singles: quarter-Kelly on the cautious estimate, 5% cap, limits per game / team / player / sport, stops at 10%
daily, 20% weekly, 30% drawdown, risk halved at 10% drawdown and if our estimates score worse than the market's.
Card tiers each have a pot (10% / 5% / 5% of bankroll) and a weekly pace (3% / 2% / 1%); losses shrink the pot and
an empty pot stops the tier, so cards can never cost more than 20% of the starting bankroll. That structure was
chosen after simulation showed any steady card habit paid from the general bankroll hits the drawdown stop.
