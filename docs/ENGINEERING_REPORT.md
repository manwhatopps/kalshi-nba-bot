# QuantSport engineering report (2026-11-25)

## The headline, measured
The system works end to end and its answer today is **NO BET on every single position**. That is the result of
the tests, not a missing feature: on 57,516 settled Kalshi game markets the team models lose to Kalshi's own price
in all nine sport-by-market groups, betting their disagreements returned -6.8%, and no player-prop model has beaten
the price either. A fund that only bets a proven edge has nothing to bet yet. Two pre-game rules sit on a watch
list at 2.2-2.4 standard errors (the bar is 3).

## What was reused
| need | reused |
|---|---|
| Kalshi market data, order book, combos, RFQ, fees | Kalshi's documented REST API (OpenAPI 3.37) through the project's own signed client `bot/kalshi.py`; combo rules read live from `/multivariate_event_collections`; Kalshi help-centre rules for combos |
| NBA data | Hugging Face `cbratkovics/nba-game-logs` (CC0); `gabriel1200/player_sheets` player tracking; `nba_api` kept as an optional home-connection source (stats.nba.com blocks cloud IPs) |
| NFL data | nflverse release files (schedules with lines, weekly player stats, play-by-play, FTN participation) |
| College football data | `sportsdataverse/cfbfastR-data` schedule/result files (CollegeFootballData content, no key needed) |
| Kalshi price history | Hugging Face `TimeSeventeen/Kalshi` (every fill), Kalshi candlesticks for college football |
| statistics / ML | numpy, scipy, pandas, statsmodels (cluster-robust logit), scikit-learn, lightgbm |
| portfolio optimisation | cvxpy (growth-optimal allocation over simulated joint outcomes) |
| simulation environment | gymnasium (`bot/sim_env.py`) |
| everything already in `bot/` | walk-forward engine, leg categories, scenarios, hedge and arbitrage tools, combo checker, slips |

Evaluated and not adopted (reasons in `docs/RESOURCES.md`): the `kalshi-python` PyPI SDK (deprecated, lacks the
current order endpoints), Riskfolio-Lib and vectorbt (built for continuous asset-return series, not same-day binary
contracts), PyMC / Optuna / XGBoost / CatBoost (no model is close enough to the price for these to matter yet),
The Odds API (paid; outside prices are reference only for a Kalshi-only fund), public "profitable bot" repos (no
verified claim).

## What was developed (`quantsport/`)
`scanner` (all three sports, both sides, rules text) · `gamedata` (results for three sports, Kalshi pre-game prices
for game markets) · `models/ratings` (points rating model, walk-forward by construction) · `models/stack` (model vs
price test and the estimate rule) · `estimate` · `mathx` (EV, break-even, Kelly variants, copula joint probability,
VaR/CVaR, drawdown scaling) · `chain` (16-step logical chain, thesis, NO BET / WAIT / ACCEPT / VETO) · `risk`
(state, exposure registry, independent auditor, portfolio Monte Carlo, Kelly portfolio) · `tiers` (four tiers,
diversification rules) · `store` (SQLite + scan snapshots) · `audit` (settlement, calibration, P&L by tier) ·
`backtest` (negative control, walk-forward, Monte Carlo of card pots) · `execution` (approval-gated order boundary)
· `dashboard` · `pipeline` (the daily run and the CIO report).

## What actually works (run on live data, 2026-11-25)
* Scan: 19,870 buyable sides in 111 games (NBA 28 games listed, NFL 14, college football 68); 106 games matched to
  the rating models.
* Settlement rule text read for every market; prices normalised to "cost to buy this side"; order-book depth walked
  for every candidate sent through the full chain.
* Probabilities for winner / spread / total / team-total markets in all three sports; props use the price plus
  graded leg categories.
* Chain: 12 candidates through all 16 steps, each recorded with its thesis; all NO BET.
* Tiers in paper mode: 3 small (1.8x / 2.2x / 2.7x), 2 diversified (5x / 10x), 1 degen (100x, 19 legs in 19
  games); no game on two cards; every card passed the auditor and Kalshi's combo rules.
* Paper book, equity curve, dashboard, performance audit (empty until games settle).
* Backtests as reported in FINDINGS section 23.

## What has been tested
`python -m pytest -q tests` -> 29 passed (executed). The QuantSport file covers: payoff, fee and rounding, EV and
break-even; binary Kelly and its variants; joint probability and correlation (copula); drawdown scaling and
VaR/CVaR; exposure registry and per-game / team / position / cash limits; stale quotes, spread and depth; daily loss
stop; one-leg-per-game, cross-card game reuse, duplicate player, pots and weekly allowances; the tier builder
(including refusing to manufacture a card and vetoing when Kalshi's combo rules cannot be loaded); the chain
(NO BET at the price, ACCEPT only on a cleared edge, WAIT price, VETO on a stale quote); settlement; approval
enforcement (no key, no terminal, changed order, wrong mode, price above limit, stale quote, reuse, expiry, exact
request body); the rating model never looking ahead; portfolio simulation and the Kelly portfolio.

## Data providers connected
Kalshi public REST (no key) · Hugging Face (NBA logs, Kalshi fills) · GitHub raw / releases (nflverse, cfbfastR,
player_sheets). All free. Nothing paid is connected.

## Keys or subscriptions needed
None to run research and paper trading. For real quotes on cards and any order: the owner's Kalshi API key, on the
owner's machine. Optional and not needed today: a CollegeFootballData key (play-by-play, opponent-adjusted
efficiency), The Odds API (reference prices), a timestamped injury feed.

## Limitations that remain
1. No proven edge. Singles: zero accepted, historically and live.
2. No injury, lineup or weather feed with timestamps: the thesis says NOT VERIFIED rather than pretending.
3. Combo quotes cannot be seen without a key; paper cards assume a fill at the walk-away payout, which real quotes
   rarely reach (buyers have been getting roughly 72-86% of fair).
4. Order sending and the RFQ path have never run against a live or demo account.
5. College football: ratings from results only (no play-by-play efficiency, recruiting, travel, weather); spreads
   and totals have no price history yet, so they are scored against the price for the winner market only; five of
   111 games on the slate could not be name-matched.
6. Closing-line value is not computed yet (needs a price capture at the start of each game).
7. NBA/NFL game-market history is the last trade before the start, not a quote a taker could always have hit.
8. Riskier assumptions are flagged in code: same-player correlation 0.35 (measured 0.2-0.4), different-game legs
   independent (measured 1.01x).
9. The approval boundary stops mistakes and replays, not someone with full access to the machine and its key file.

## How to start, run, review, paper trade
See `docs/OPERATING_GUIDE.md`. In short: `python -m quantsport daily --mode paper`, read the printed report or
`data/quantsport/dashboard.html`, `python -m quantsport settle` and `performance` the next day.

## Before live trading is enabled
Eligibility and Kalshi's developer terms checked by the owner; a trade-only key on the owner's machine; the
sending and quote paths exercised on Kalshi's demo exchange; the approval key created at a terminal; a paper record
long enough to judge (hundreds of settled positions, not weeks); and, above all, at least one rule that clears the
fund's own bar. None of these is done.
