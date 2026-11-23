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

## 4. Which mix of legs is the sweet spot?  Mid-priced legs, few of them.
Per leg at the price you pay (`python -m bot mixes`), and what is left of fair value after stacking
that kind of leg to a ~10x payout:

| leg price | legs | paid | hit | legs needed for 10x | fair value kept at 10x |
|---|---|---|---|---|---|
| 35-50c | 3,400 | 0.420 | 0.407 | 2.7 | 92% |
| 50-60c | 2,020 | 0.545 | 0.531 | 3.8 | 91% |
| 60-70c | 2,115 | 0.645 | 0.635 | 5.3 | 92% |
| 70-80c | 2,234 | 0.745 | 0.739 | 7.8 | 94% |
| 80-90c | 2,684 | 0.848 | 0.831 | 14.0 | 76% |
| 90-95c | 1,587 | 0.920 | 0.907 | 27.5 | 69% |
| 95-99c | 1,274 | 0.968 | 0.936 | 70.0 | 10% |

Every band loses 1-3 cents per leg; expensive legs just need far more legs to reach the same payout,
so they pay that cost many more times. Whole cards dealt at random from settled nights agree:
50-70c cards at ~5x and ~10x hit 16.1% and 7.6% of the time against 15.4% and 7.7% needed (fair
within noise); bigger payouts kept 60-95% with wide error bars.

The combo shared at kalshi.com/s/m/KXMVECROSSCATEGORY-S2026CW5OA5GGJYZDICS3/SOLPKKFPB is this kind
of mix (8 NHL legs at 47-79c). Its legs are worth 57x at mid prices; it last traded at 50x, i.e.
buyers paid ~14% over the legs' value. `python -m bot mix` prints that ceiling for every card.

## 5. "Will it hit?" — does the simulation know more than the price?  No.
9,530 settled points props, grouped by how the 20,000-run simulation compared with Kalshi's price:

| simulation vs price | legs | Kalshi price | simulation said | actually hit |
|---|---|---|---|---|
| much lower | 979 | 51.6% | 35.0% | 50.1% |
| lower | 1,367 | 42.5% | 35.9% | 43.3% |
| agree | 3,814 | 34.8% | 34.9% | 34.0% |
| higher | 2,087 | 37.7% | 44.3% | 38.1% |
| much higher | 1,283 | 35.9% | 51.3% | 35.9% |

Legs hit at the rate Kalshi's price implies whatever the simulation says; last-10-game hit rates
show the same thing. The price is the best "will it hit" number available, so the card builders
use it (plus each leg type's settled hit rate) rather than the simulation.

## 6. Rotobot and copy trading
* Rotobot publicly describes simulating each game 10,000 times and reporting how often each prop
  clears (no published track record found). `python -m bot sims` does the same with 20,000
  correlated runs per game; it was rebuilt from the public description, not from their code.
* Kalshi's public leaderboard is readable, but all 12 top Sports traders checked hide their
  holdings and only ~10% of trades in the public feed carry a name (0 of 91 named trades tonight
  came from the top 100). Kalshi's own research (Oct 2026) found profit and forecasting skill are
  largely unrelated. `python -m bot social` shows what is visible; it uses an undocumented website
  API, so it is for occasional manual use and is not part of the daily run.

## Bottom line
A 50-leg favorites combo is the least efficient shape tested. The most efficient is 3-8 legs priced
35-80c on tight books, bought only when the quote is within ~10% of the legs' mid value: that is
close to a fair bet with a 5-35x payout. Nothing tested so far has a positive expected value as a
combo buyer. The bot keeps paper-trading every strategy daily so this can be re-judged on more
data, and it will not place real money on a strategy whose backtest is negative.
