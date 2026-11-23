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

## 7. Every fill this season: who actually made money?
Source: all 45.2M NBA/NFL fills on Kalshi from 2026-10-20 to 2026-11-21 (27.8M NBA fills, 248 games,
36 market types), each one graded against the settled result (`python -m bot trades`). Every fill is
two real bets: the taker who paid the quote and the maker whose resting order was hit.
ROI after fees, every fill counted once, errors clustered by game.

| NBA, price paid | taker, pre-game (last 3h) | taker, in-game | maker, in-game |
|---|---|---|---|
| under 10c | -45% | -46% | -24% |
| 10-20c | -34% | -21% | -7% |
| 35-50c | +3% (noise) | -1% | +2% (noise) |
| 50-65c | -4% | -5.3% | -2% |
| 80-90c | +4% (noise) | -0.1% | +2% (noise) |
| 90-95c | +3.3% | +0.7% (noise) | +2.5% |
| 95-99c | -0.6% | **+0.7%** | **+1.9%** |
| 99c | -0.2% | +0.3% | +0.7% |

* Long shots are the worst bet on the board for everyone: under-10c fills lost 21-46%.
* In-game fills at 95-99c were positive for both sides, 5-9 standard errors from zero, in both
  halves of the season. But that is an average over fills, and blowouts produce thousands of safe
  fills late in games. See the next section for what a followable rule earned.

## 8. A rule you can follow: buy once, the first time a market touches the price in-game

| NBA market | threshold | bets | games | avg price | won | losses | ROI after fee |
|---|---|---|---|---|---|---|---|
| moneyline | 90c | 474 | 216 | 0.901 | 93.0% | 33 | +2.6% (+/-1.7) |
| moneyline | 95c | 426 | 203 | 0.950 | 97.9% | 9 | **+2.7% (+/-1.0)** |
| moneyline | 97c | 398 | 193 | 0.970 | 99.0% | 4 | +1.8% (+/-0.7) |
| spreads | 95c | 28,520 | 236 | 0.966 | 97.1% | 821 | +0.3% (+/-0.2) |
| totals | 95c | 40,926 | 236 | 0.965 | 96.6% | 1,374 | -0.1% (+/-0.2) |
| player props | 95c | 68,187 | 236 | 0.967 | 96.6% | 2,288 | -0.2% (+/-0.1) |
| all markets | 95c | 145,879 | 236 | 0.966 | 96.7% | 4,831 | -0.1% (+/-0.1) |

(+/- is one standard error. Moneyline bets count each game twice: YES one team / NO the other.)

* Across all markets the followable rule is flat to slightly negative: the fill-level "edge" in
  section 7 was mostly garbage-time fills.
* One segment survives: **NBA in-game moneylines at 95c+**. A team priced 95c won 97.9% of the
  time. Positive in both halves of the season (+1.4%, +3.9%), but it rests on ~200 games and 9
  losses, and it is the best of 20 segments tested, so treat it as a lead, not a fact.
  NFL (59 games) shows nothing reliable.
* `python -m bot watch` now paper-trades exactly this rule live, at the real ask, every game night.

## 9. Bankroll reality (`python -m bot bankroll`)
$100 bankroll, $2 per card, 200 cards, using the measured per-leg cost plus a 10% maker cut:
5x cards from mid-priced legs end ahead 13% of the time; 25x cards 18%; 95c+ stacks 0%. Kelly sizing
says $0 on every combo type (none is +EV). It takes ~500 cards at 5x, ~1,200 at 10x and ~12,700 at
100x before results can tell a winning card strategy from a losing one, so a handful of hits or
misses proves nothing either way.

## 10. Segment scan with multiple-testing control (`python -m bot edges`)
42 leg types (market x side x price band x snapshot) tested on quote data with a discovery half and
an untouched confirmation half and Benjamini-Hochberg false-discovery control: 0 passed.

## Bottom line
* Buying combos is at best a fair bet. The efficient shape is 3-8 legs priced 35-80c on tight books,
  bought only when the quote is within ~10% of the legs' mid value. 50 legs at 90c+ is the least
  efficient shape tested. Long shots under 20c are the worst bet on the board.
* Kalshi's price is a better "will it hit" predictor than any model or simulation tested.
* The one lead with positive expected value is a single bet, not a combo: NBA in-game moneyline
  favorites at 95c+ (+2.7% +/- 1.0% over ~200 games). It is being paper-traded live.
* No real money has been placed. Nothing here justifies real stakes on combos; the moneyline rule
  needs more games (and a live paper record) before it does.
