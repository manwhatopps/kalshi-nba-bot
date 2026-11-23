# What the data says so far (2026-11-23)

Tested on Kalshi's own settled NBA markets, 2026-10-20 to 2026-11-22 (33 nights, 248 games).
Loaded so far: moneylines + player points (13,635 markets, 23,374 priced legs). Rebounds, assists,
threes, spreads, totals, other props and NFL are still downloading (Kalshi rate limit); every
number below gets re-run automatically as they arrive (`scripts/daily.sh`).

## 1. Do our models beat Kalshi's price?  No.
8,359 player-points legs scored out-of-sample over 23 nights (`python -m bot models`):

| predictor | Brier (lower = better) |
|---|---|
| Kalshi pre-game mid | **0.1640** |
| stats model (negative binomial) | 0.1713 |
| recent hit-rate model | 0.1742 |
| gradient-boosted model | 0.1727 |
| walk-forward stack of all four | 0.1643 |

The stacker puts 0.96 of its weight on Kalshi's price and 0.05 on the model. Betting the stacker
against real quotes after fees: 423 bets, ROI -6.4% (95% interval +/- 9.2%). The live scanner now uses these
learned weights, so it reports no value legs unless the evidence changes.

## 2. Does "many legs at 90c+" work?  Not at the prices you can actually pay.
Win rate vs price, pre-game (`python -m bot favorites`):

| leg price | n | paid | won | after fee |
|---|---|---|---|---|
| 0.90-0.93 | 1,213 | 0.914 | 0.904 | -1.6% |
| 0.93-0.95 | 374 | 0.938 | 0.920 | -2.3% |
| 0.95-0.97 | 605 | 0.955 | 0.931 | -2.7% |
| 0.97-0.99 | 460 | 0.975 | 0.939 | -3.7% |
| 0.99+ | 209 | 0.990 | 0.943 | -4.8% |

* The loss is the bid-ask spread on thin prop books, not bad odds: at the MID price the same legs
  are fairly priced (edges within +/-1-2%, not significant).
* Whole cards (150 random legal cards per night, payout = product of leg prices, the best case):
  10 legs at 0.90+ hit 43.6% of the time vs 54.0% needed -> ROI -21%. At mid prices the same cards
  are roughly break-even (-10%, standard error 9.5%).
* A 50-leg card multiplies the per-leg cost 50 times: 2% per leg leaves 0.98^50 = 36% of fair value.
* Independent research agrees: on 72M Kalshi trades, takers at 90-99c earn -0.3% before fees and
  makers +1.8% (Becker 2025); on 8.9M Kalshi combos, makers keep about 2-3c per combo and favorite
  parlays win less often than the product of their leg prices (jsteng19/kalshi-combos-research).

## 3. Anything that looks like an edge?
* In-game, legs with a MID of 0.95-0.99 won ~2 points more often than priced (n=814 and 825 at two
  snapshots, about 4 standard errors) — but buying at the ask erases it. It can only be captured
  with resting limit orders (being the maker), which combos do not allow.
* Pre-game YES legs at 0.90-0.95 on points props: +2.7% at mid (n=118, +/-4%). Too small to trust;
  re-check once rebounds/assists/threes history is in.

## Bottom line
A 50-leg favorites combo is a fair-to-slightly-unfair lottery ticket: it hits about as rarely as
its payout implies, minus trading costs. Nothing tested so far has a positive expected value as a
combo buyer. The bot keeps paper-trading every strategy daily so this can be re-judged on more
data, and it will not place real money on a strategy whose backtest is negative.
