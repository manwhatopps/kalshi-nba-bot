# What the data says so far (2026-11-23)

> **Read sections 12, 14 and 15 first.** The moneyline "edge" found on this season's games (sections 8
> and 11) lost money on 1,268 games from Kalshi's prior-season archive. A player-prop rule (section
> 14) is positive in both seasons separately and is the fund's one graded candidate.

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

## 11. The fund: how the bankroll is allocated (`python -m bot fund`)
The bot is run as one bankroll with three sleeves, not as a pick list.

* **Core** (default 50%): single bets on graded edges, counted one bet per game.
  | edge | bets | lost | won | avg price | break-even | per $1 | grade |
  |---|---|---|---|---|---|---|---|
  | NBA in-game moneyline, first touch of 95c | 203 | 5 | 97.5% | 95.1c | 95.4% | +2.3% | B candidate |
  | NBA in-game moneyline, first touch of 97c | 193 | 2 | 99.0% | 97.0c | 97.2% | +1.8% | B candidate |
  | NBA in-game moneyline, first touch of 90c | 216 | 17 | 92.1% | 90.1c | 90.7% | +1.5% | C unproven |
  | NFL in-game moneyline, first touch of 95c | 36 | 2 | 94.4% | 95.0c | 95.3% | -0.9% | C unproven |

  Grades: A = confirmed on games after discovery; B = positive even at the low end of its error
  range but not yet confirmed; C = no money. Kelly on the 97c rule's measured win rate is 63% of
  bankroll; on the low end of its error range it is 19%. The plan stakes the cautious figure per
  bet, capped so the whole sleeve covers the night's expected triggers (the rule fires in ~82% of
  games), instead of putting the sleeve on one game: same expected profit, far less risk.
* **Degen** (default two $5 tickets aiming at $1,000 and $10,000): fewest fairly-priced legs that
  reach the payout, with a minimum acceptable quote. Fixed cost, never sized up.
* **Reserve**: the rest, never deployed.

Simulated over 60 game nights from $100 (lottery tickets assumed to return half of fair value):
with the edge as measured, the core alone ends at a median of about $171; adding $10 of tickets
every night ends at about $0 in every scenario, because $600 of tickets cannot be carried by a core
that makes ~$1 a night. Tickets bought only out of profits keep the fund near $100 with a 1-2%
chance of a big hit.

## 12. The archive test: the core edge fails on last season's games
Kalshi's archive has every NBA moneyline back to the April 2025 playoffs (1,449 games) and NFL back
to August 2025 (414 games). All fills for those markets were pulled (56.9M) and the same first-touch
rule replayed, one bet per game.

| rule | where it was found (this season) | independent games (archive) | verdict |
|---|---|---|---|
| NBA first touch of 95c | 203 bets, won 97.5%, +2.3% | 1,268 bets, 77 lost, won 93.9% vs 95.4% break-even, **-1.6%** | failed |
| NBA first touch of 97c | 193 bets, won 99.0%, +1.8% | 1,225 bets, 48 lost, won 96.1% vs 97.3%, **-1.2%** | failed |
| NBA first touch of 90c | 216 bets, won 92.1%, +1.5% | 1,348 bets, 154 lost, won 88.6% vs 90.9%, **-2.6%** | failed |
| NFL first touch of 95c | 36 bets, won 94.4%, -0.9% | 198 bets, 17 lost, won 91.4% vs 95.5%, **-4.2%** | failed |

The 2025-26 regular season alone (1,114 games) gave -0.8% at 95c; both playoff runs were worse.
This season's +2.3% was the best of 20 segments on 203 games - a lucky sample, as the archive
shows. Edges are now graded on games they were NOT found on (`bot/fund.py`); all four grade
"C failed", so the fund's core sleeve holds cash.

Other patterns re-checked on the archive, one bet per game: pre-game NBA favorites at 88-99c won
93.6% of 125 games against a 92.0% break-even (+1.7%, well inside noise); pre-game favorites at
50-70c lost 7.7% as buyers. Fill-level, resting orders filled at 95-99c in-game were positive in
both seasons (+1.8% archive, +1.9% this season), but that is an average over fills, not yet a rule
with a fill simulation - the same trap section 8 fell into.

## 13. The simulator (`python -m bot sim`)
`bot/sim_env.py` is a gymnasium environment (passes `check_env`) that replays 1,471 real NBA games
on 278 nights: entries in the order they happened, money locked until each game ends, reward = log
growth. Policies compared from $100:

| policy | this season's nights (60, reshuffled) | last season's nights (60, reshuffled) | everything in order (278 nights) |
|---|---|---|---|
| stay in cash | $100 | $100 | $100 |
| 50% of bankroll on one bet a night | $24 | $24 | $1 |
| 20% per trigger | $220 | $39 | $3 |
| 6% per trigger | $164 | $72 | $33 |
| 2% per trigger | $118 | $91 | $72 |
| evidence-based Kelly (bets only once its own record clears break-even) | $100 | $100 | $100 |
| 6% per trigger + $10 of lottery tickets nightly | $1 | $1 | $1 |

(Median ending bankroll.) The same policy that turns $100 into $220 on this season's nights turns
it into $39 on last season's: sizing cannot rescue a rule without an edge. The evidence-based
policy never bet on the archive and so never lost. Concentrating half the bankroll on one bet a
night loses even on the favourable sample, because a single miss halves the fund.

## 14. Player props across two seasons (`python -m bot props`)
Every prop fill from Kalshi's archive (2.7M NBA fills over 1,091 games, 4.6M NFL over 343) and from
this season (5.0M NBA and 2.9M NFL prop fills) was graded. Followable version: buy once, the first
time that side trades in a price band during the game; return after fees, errors clustered by game.

| in-game, all prop types | last season | this season |
|---|---|---|
| NBA OVER, 35-50c / 50-65c / 65-80c / 80-95c | -21.9% / -18.8% / -16.9% / -11.3% | -6.5% / -4.6% / -2.9% / -2.9% |
| NBA UNDER, same bands | +12.0% / +9.1% / +3.0% / -0.1% | -3.5% / -1.5% / -1.7% / -1.5% |
| NFL OVER, same bands | -20.3% / -18.6% / -19.8% / -19.3% | -12.4% / -8.9% / -7.4% / -3.7% |
| NFL UNDER, same bands | -1.0% / -0.9% / -3.5% / -2.4% | +2.4% / +1.1% / -1.3% / -0.6% |

* **Buying prop overs during a game loses in both seasons, both sports, every price band and every
  prop type** (16,000-30,000 bets per cell, 3 to 18 standard errors). This is the firmest finding in
  the project, and it is the "will he get there?" chase.
* Unders in general were profitable last season and are not this season (10x the volume, tighter
  prices). Pre-game, overs and unders are both roughly fair before fees and lose after them; props
  priced 95-99c on either side hit ~4 points less often than they cost.
* By prop type, the under held up in both seasons for double-doubles (below), triple-doubles
  (+3.8%, +21.5%), steals (+7.4%, +3.4%) and blocks (+8.8%, +2.1%); the last three are small or
  not individually significant.

**The graded candidate: NBA in-game double-double UNDER, first fill at 50-80c, one bet per market.**

| | bets | games | won | break-even | return after fee |
|---|---|---|---|---|---|
| last season | 691 | 399 | 74.0% | 66.3% | **+11.5%** (t = 4.7) |
| this season | 265 | 134 | 78.1% | 68.0% | **+14.9%** (t = 3.5) |

Positive in 8 of 9 months and in all four price bands in both seasons; the over on the same
markets lost 38% and 19%. Grade B (found using both seasons; not yet tested on games after
discovery). Limits: these are thin markets - a typical one saw about $5 of unders bought at those
prices and the first fill was usually one contract - so it suits a small fund and nothing larger.
In the simulator, with $5 fillable per bet, $100 becomes $638 over the 172-night history ($315 with
$2 per bet); without a liquidity cap the simulator shows absurd growth, which is why the cap exists.
`python -m bot watch` paper-trades the rule live and records the size on offer at each entry.

## 15. Walk-forward backtest of the fund's logic (`python -m bot walk`)
A backtest of the decision process itself, on data already downloaded, with no hindsight. On each of
345 nights (April 2025 - November 2026) the bot sees only earlier nights. From 142 candidate rules
(prop type x over/under x price band, plus moneyline first-touch rules; 422,900 candidate bets) it
activates a rule only if its past record has 150+ bets over 60+ games and a return that is positive
even 3 standard errors below its estimate, stakes a quarter of the cautious Kelly bet (at most $5
per bet), and takes the real result.

| bar for activating a rule | rules used | bets | return on stakes | worst drawdown |
|---|---|---|---|---|
| any positive record | 66 | 47,311 | +1.5% | $3,368 |
| positive by 1.64 std errors | 31 | 29,178 | +2.4% | $2,169 |
| positive by 3 std errors | 12 | 16,187 | +3.5% | $738 |
| positive by 4 std errors | 8 | 6,935 | +3.5% | $211 |

* A stricter bar means fewer rules, a better return per dollar and much smaller drawdowns. At the
  loose bar 14 of 31 activated rules lost money afterwards.
* **By season, at the 3-standard-error bar: last season +9.7% on $27,749 staked; this season -1.1% on
  $36,790.** The logic found the unders on points, threes, rebounds and assists by itself last
  season and they paid; this season it kept betting them on their lifetime record after they stopped
  working (points under 35-50c: -4.4%; rebounds under: -5.9%).
* A recency check (the rule must also be profitable over its last 20 nights) only moves this season
  from -1.1% to between -1.2% and +0.7%. It is now part of the logic, but it is a patch, not a cure.
* The dollar totals assume $5 fills on every bet, about 47 bets a night. With $1 per bet the whole
  history earns $585; with no cap the backtest shows $29,000, which is not achievable.

**The double-double under, without hindsight.** The logic activated it on 2026-04-01. Counting only
bets after that date: 369 bets over 200 games, won 64.8% against a 55.3% average cost, **+17.1%
(t = 3.8)**; positive in the rest of last season and in this season. In the fund's registry (one
bet per market, first fill at 35-80c): 449 bets before activation at +12.1%, 605 bets after at
+15.6%. Graded A; stakes are capped at $5 per bet for liquidity.

What the logic would bet tonight: double-double unders at 35-80c, and (tracked without money) NFL
first-touchdown unders at 65-80c. The broad unders no longer pass.

## 16. Degen mode: what a big-payout card should be built from (`python -m bot degen`)
Pre-game, one leg per market at the last price a buyer paid in the final 3 hours, legs priced 50-75c:

| | last season | this season |
|---|---|---|
| NBA overs: paid / hit | 60.8c / 57.3% | 61.6c / 60.5% |
| NBA unders: paid / hit | 60.2c / 61.4% | 60.9c / 60.6% |
| NFL overs: paid / hit | 60.1c / 55.0% | 60.9c / 56.4% |
| NFL unders: paid / hit | 60.9c / 60.3% | 60.7c / 61.6% |

Overs hit 1-5 points less often than they cost; unders hit about what they cost. Of 145 pre-game
prop angles, the 17 where buyers beat the price in both seasons are all unders (none individually
significant). Random legal lottery cards dealt from each night's real legs (216 nights):

| card built from | ~50x, last / this season | ~200x | ~1,000x |
|---|---|---|---|
| overs | kept 82% / 86% of fair value | 63% / 103% | 46% / 65% |
| unders | kept 92% / 99% | 62% / 130% | 21% / 260% |

At ~50x unders are clearly the better build. At 200x and 1,000x cards hit so rarely that two seasons
cannot rank anything (the same recipe scores 0.2 and 2.6). A narrower "double-double, triple-double,
blocks, steals, threes unders only" card did NOT beat plain unders and is not used.

Degen mode therefore builds all-under cards from 50-75c legs on tight books, one leg per event, with
a 25% cushion between the fair payout and the payout the ticket needs. **No card has a positive
expected value.** Honest odds for the default nightly set ($10 -> $1,000, $5 -> $1,000, $5 -> $5,000):
about a 1.4% chance per night that one hits, about 9% over a week for $140 of tickets. From a $100
bankroll the chance of reaching $1,000 in a week cannot exceed about 10% by any method (at fair odds
it equals bankroll / target), and it is lower after the market maker's cut.

## 17. Play styles, matchups and the "logical chain" (`python -m bot styles`, `python -m bot matchup`)
Built from player box scores, walk-forward (each game sees only earlier games):

* **Team style profiles for all 30 teams** over their previous 15 games: pace, offensive and defensive
  rating, three-point rate, free-throw rate, assist rate, offensive-rebound rate, turnover rate, and
  the same for what the defense allows. Box scores cannot see a scheme directly; they see what it
  concedes (threes allowed, assisted baskets allowed, fouls, offensive rebounds allowed, turnovers
  forced). Example tags this week: Houston, Memphis, Portland and Utah allow few assisted baskets
  (the signature of switching); Boston plays slow, shoots and concedes threes; Chicago, Milwaukee and
  Washington are weak on the defensive glass.
* **Six player roles** from k-means on per-36 production (primary scorer, playmaking guard, spot-up
  shooter, versatile big, rim big, role player), standing in for positions.
* **Defense-vs-role**: how much each defense has allowed each role, per minute, relative to those
  players' own norms, over its previous 20 games.

**Do these add anything to Kalshi's price? No.** On 68,262 settled NBA prop markets with a pre-game
price, regressing the result on the price plus each factor (errors clustered by game):

| factor | last season | this season |
|---|---|---|
| Kalshi's price (1.00 = fully trusted) | 1.01 | 1.03 |
| recent form vs the line | +0.079 (z 3.8) | -0.008 (z -0.4) |
| defense-vs-role | +0.029 (z 1.6) | +0.015 (z 0.7) |
| opponent defensive rating | z 0.3 | z 0.3 |
| expected pace | z 0.7 | z -0.2 |
| minutes trend | z -0.5 | z -0.2 |
| team on a back-to-back | -0.059 (z -2.3) | -0.012 (z -0.4) |
| opponent on a back-to-back, home, blowout risk | all under z 1.5 | all under z 1.5 |

Nothing is distinguishable from zero in both seasons. Fitted on last season and scored on this one,
price + factors (log-loss 0.5618) is slightly worse than the price alone (0.5606). Scheme proxies
matched to the stat they should affect (threes vs threes allowed, rebounds vs glass weakness, assists
vs assisted looks allowed) show the same: one was significant last season, none this season. Legs
where the whole chain agrees with the under did no better this season than unders in general
(+0.1 to +0.9 points vs +0.9).

**Sharper scheme data, same answer.** Per-game tracking for every team (gabriel1200/player_sheets:
rim, mid-range and three-point shares of shots, catch-and-shoot and pull-up attempts, wide-open
threes, drives, paint and post touches, potential assists, seconds per touch) was turned into what
each defense allowed over its previous 15 games, and into scheme labels: San Antonio, Portland and
Utah look drop-style (mid-range and pull-ups conceded, threes taken away); Boston, the Lakers and
Oklahoma City pack the paint and concede threes; Brooklyn, Milwaukee and New York allow
catch-and-shoot threes and ball movement (zone-like); Minnesota, Detroit and Orlando force isolation
(switch-style). Thirteen pre-chosen tests matched each measure to the stat it should affect; none
is distinguishable from zero in both seasons, and price + all 13 scores worse out of sample (0.5613)
than the price alone (0.5606).

One lean has the same sign in both seasons and a mechanism: **assist props against isolation-forcing
defenses**. Assist unders beat their price by 2.2 points last season and 0.8 this season against
the top third of defenses by seconds per touch, and hit 1.8 and 1.1 points LESS than priced against
the bottom third (pooled z = -3.0 on 10,742 assist markets, right at the bar for 13 tests). It is
smaller than Kalshi's fee, so it is a tie-breaker in the chain, not a rule to bet.

What does hold pre-game: **unders as a group beat their price by 2.8 points last season and 0.9 this
season** - the structural over/under tilt, not matchup insight.

So the chain is context, not an edge. Degen cards print it for every leg (`--why`): recent form vs
the line, minutes trend, role and what this defense gives that role, pace, opponent defense, rest.
It is used only to drop legs where it clearly points to the over and to order otherwise-equal legs.

## 18. From picks to slips (`python -m bot slip`, `python -m bot combos`)

A pick becomes a bet only when it names the exact market and side, the size, the worst price worth
paying, how long that holds, and passes Kalshi's own rules. Measured on 2026-11-23:

**Kalshi's combo rules, read from its combo collection (`/multivariate_event_collections`).**
All three open collections carry the same 2,538 events and the same rules.
* Player props (points, rebounds, assists, threes, steals, blocks, PRA, double-double) can be
  combined on either side, and any number of legs may come from one prop group. An all-UNDER card
  is therefore a legal combo.
* Moneyline, first-half and first-quarter winners are YES-only. Spreads and totals take both sides
  but one leg per game.
* A combo needs at least 2 legs. Creating one needs the owner's API key; the price comes from a
  request-for-quote to market makers. The old "lookup" endpoint was removed in August 2026.
* The earlier card builder allowed one leg per event. That was stricter than Kalshi: it now uses
  Kalshi's own limit and keeps one leg per player instead, so legs stay close to independent.

**What 118,133 real combos look like (random sample, created Oct 24 to Nov 22).**
11% of legs were bought as NO; 2.1% of combos were all NO; 60% took two or more legs from one
market group (up to 26). Hit rate falls steadily with leg count: 29% at 2 legs, 13% at 5, 6.7% at
9-10, 4.8% at 11-15, 2.5% at 16-25, 1.3% above 25.

**Same-game legs move together.** For 22,754 combos whose legs are all props with a pre-game
price, the hit rate was compared with the product of the legs' own prices:

| legs per game | combos | hit rate | product of leg prices | ratio |
|---|---|---|---|---|
| 1 | 2,776 | 15.4% | 15.3% | 1.01 |
| 1 to 1.5 | 2,580 | 11.3% | 10.6% | 1.06 |
| 1.5 to 2 | 3,484 | 15.3% | 13.4% | 1.14 |
| 2 to 3 | 3,630 | 14.0% | 12.1% | 1.15 |
| 3 to 5 | 5,067 | 12.8% | 9.8% | 1.30 |
| over 5 | 5,214 | 8.5% | 5.1% | 1.67 |

With one leg per game the product is right. Stacking a game makes the combo hit more often than
the product says, mostly because buyers stack legs that rise together. **This is not an edge by
itself**: the market makers who quote combos can price that in. It does mean the "1 in N" printed
on a slip, which treats legs as independent, is close for our cards (different players, at most
3 per game) and would be too pessimistic for a one-game stack. It also means the earlier
"sold at 15% of fair value" example was measured against an independence product and is only
valid if that combo's legs were unrelated.

**What buyers actually paid.** Kalshi publishes every fill of a combo (`/markets/trades`), one
call per combo. A sample of 2,200 (220 per leg-count bucket) is being fetched into
`data/combo_prices.parquet`; `python -m bot combos` prints hit rate against cost by leg count.
Only 50 were priced when this was written, far too few to read. An unverified third-party study
(jsteng19/kalshi-combos-research) reports takers overpaying by about 3 cents per combo on average.

**Pricing audit against Kalshi's live order book (2026-11-23, 6:27 pm ET).** Kalshi's real-time feed
is a WebSocket that needs an API key even for public channels, so without a key the bot reads the
public order-book endpoint (`GET /markets/orderbooks`, 100 markets per call, every resting bid with size).
* The best bid and ask the bot had been using (from the market list) matched the order book on all
  53 markets in that night's slips. Prices were right.
* Sizing was not: contracts were counted at the limit price, so a slip could show 6 contracts where
  the book filled 7 for the same $5. Slips now walk the real book level by level (`bot/book.py`) and
  report how many contracts are offered inside the limit.
* The combo walk-away line ignored the fee. Combos carry the same taker fee as single markets
  (7% x price x (1 - price), series fee type `quadratic_with_combo_maker_fees`), so "$10 returns
  $1,000" needs a quoted price of about 0.935c, not 1.00c. Corrected; the dollar walk-away line
  ("$1,000 or more for $10") was already right because the payout Kalshi shows is after the fee.
* Fees are charged per order and rounded up to the cent for retail accounts (Kalshi "Fee Rounding").
* All prop series checked are fee type `quadratic`, multiplier 1.

**A live example of a big card's price (2026-11-23).** A 24-leg combo of game winners the owner
shared (22 college football games, one NBA, one NFL, all different games, so the legs are
independent): the legs' own order-book prices multiply to 0.0075%, 1 in about 13,300. It was selling
at 0.05-0.08c a contract, which pays about 1,335x after the fee. Buyers were getting 10 cents of
value per dollar; 73 purchases had paid $117 in total. Two underdog legs (Cincinnati over BYU at 16c,
Michigan over Ohio State at 24c) carry most of the risk. Past a few thousand to one the price stops
falling as legs are added, so every extra leg is given away. `python -m bot combo <link>` runs this
check on any combo.

**What a slip contains.**
* Single bets (core sleeve): market, BUY NO, contract count for the fund's stake, a limit 2c above
  the price seen and never past the 80c edge of the tested range, when to place it (the rule is
  in-game), the rule's own record in that price band, and the order in the form Kalshi's current
  order endpoint takes (`POST /portfolio/events/orders`; the older `/portfolio/orders` now returns 410).
* Lottery cards: legs in the form the combo endpoint takes, a rule check, an expiry (first tip-off),
  and a walk-away payout. `slip --check` re-prices everything before placing.
* `slip --quote` (needs the key) creates each combo and prints the real quote next to the walk-away
  line. `--live` accepts only above that line and inside the stake caps. Not yet run against an
  account: there is no key on this machine.

## 19. Scenarios: player type against defensive style (`python -m bot scenarios`)

The question: does a certain kind of player have a real advantage against a certain kind of defense
(a pull-up three shooter against drop coverage, a receiver against man or zone), and can it be bet?
Two separate tests for every scenario: is the advantage **real** in the box score, and is it already
**priced** by Kalshi. All walk-forward.

**NBA.** Player-level tracking for every game of 2025-26 and 2026-27 (gabriel1200/player_sheets:
pull-up and catch-and-shoot threes, shots by zone, drives, post touches, potential assists).
Each player's shot diet is his previous 15 games. What a defense concedes is what its opponents
took divided by what those same players normally take, over its previous 15 games, so it is
adjusted for who it played. 23,073 player-games and 67,650 settled prop markets. Each row compares
the same kind of player against the top third of defenses on the measure vs the bottom third.

| scenario | what happened (per minute, vs own norm) | real? | overs vs Kalshi's price |
|---|---|---|---|
| pull-up three shooter vs drop-style defense | pull-up threes +4.7% (z 2.1), but points -4.3% (z -2.9), threes made -9.1% (z -3.3) | more looks, FEWER points | -1.1 pts (z -0.6): priced |
| pull-up three shooter vs defense conceding pull-up threes | pull-up threes +4.7%, points -0.9% (z -0.6) | no | priced |
| spot-up shooter vs defense conceding catch-and-shoot threes | threes made +18.8% (z 5.3), points +7.0% (z 3.5) | yes, strong | +1.5 pts (z 0.6): priced |
| rim finisher vs defense conceding shots at the rim | rim shots +16.8% (z 6.3), points +7.4% (z 3.2) | yes | +4.3 pts (z 1.5): not established |
| playmaker vs defense conceding assisted looks (vs switching) | assists +6.7% (z 3.5) | yes | +3.9 pts (z 1.7): not established |
| driver vs defense conceding drives | drives +3.0% (z 3.0), points -2.3% (z -2.1) | more drives, not more points | priced |
| post scorer vs defense conceding post touches | post touches +16.7%, points +4.1% (z 1.4) | not proven | overs did WORSE, -8.2 pts (z -2.4), both seasons |

What this says:
* **The drop-coverage idea is half right.** Against drop-style defenses pull-up shooters do get more
  pull-up threes. They also score less and make fewer threes. Drop coverage concedes the pull-up on
  purpose because it is the least efficient shot, and the defenses that play this way concede 2.4%
  fewer points overall. More looks is not more points.
* **Some matchup advantages are real and large**: spot-up shooters against defenses that help off
  shooters, rim finishers against soft rims, playmakers against defenses that allow ball movement.
* **Kalshi's price already contains them.** No scenario shows overs beating the price with |z| > 2
  in both seasons. Shot-diet exposure for every player at once: the favourable third out-produced
  the unfavourable third by 7.2% in points last season (z 7.5) and 1.1% this season (z 0.6). After
  the price it added a little last season (z 2.3) and nothing this season (z 0.0), so it is not
  established. Rebounds: +4.0% and +7.8% (real in both seasons), nothing after the price.
* The one result against the price runs the wrong way for the story: post scorers in "good post
  matchups" were overpriced in both seasons. With seven scenarios tested, one z of -2.4 is weak
  evidence; it is recorded, not bet.

**NFL.** Every charted dropback of 2024 and 2025 (nflverse, coverage charted by FTN): the defense's
man or zone call, who was on the field, who caught what. 2,487 receiver-games in 2025 with 150+
charted routes of history, and 6,375 settled Kalshi receiving markets.
* A receiver's man-vs-zone split is mostly noise: first half of 2025 against second half,
  correlation +0.20 across 103 receivers.
* A defense's coverage is moderately predictable: previous-8-game man share against what it then
  played, correlation +0.43.
* Man-beaters did not do better against man-heavy defenses (-9.1% vs zone-heavy, z -1.4, the wrong
  direction). Zone-beaters against zone-heavy: +9.8% (z 1.3). Coverage exposure for all receivers:
  +2.0% (z 0.5). Nothing is distinguishable from zero.
* Against Kalshi's price, overs in the "favourable coverage" third did WORSE than in the
  unfavourable third (-4.5 points, z -1.7; -0.142 per standard deviation after the price, z -2.9).
  Betting the man/zone story would have lost in 2025. One season, so not a rule either way.
* Coverage charting for 2026 is published only after the season, so none of this can be applied
  to this year's games.

**How it is used.** Every NBA leg now prints its scenario (`slip --why`, the slip page): the player's
type from his own shot diet, where tonight's defense ranks on the measure that matters for him, what
happened historically, and whether it is priced. A scenario counts toward dropping or keeping a leg
only if the box-score effect clears 3 standard errors with the same sign in both seasons. It is a
reason, not an edge: the measured edge is still zero after Kalshi's price.

## 20. Pre-game only (`python -m bot walk --pre`)

Owner's decision, 2026-11-23: every bet is placed before the game; no live betting and no live
watcher. The fund's logic was therefore re-tested on pre-game prices alone: 188,029 candidate
prop bets over 267 nights (Sep 2025 to Nov 2026), one per market and side at the last price a taker
paid in the three hours before tip-off, 190 rules (prop type x over/under x price band).

| bar a rule must clear before it gets money | rules used | bets | last season | this season | $100 becomes |
|---|---|---|---|---|---|
| any positive record | 49 | 6,470 | -1.8% | -2.7% | $31 |
| positive by 1.64 std errors | 2 | 64 | -61.3% | -9.6% | $99 |
| positive by 3 std errors (the fund's bar) | 0 | 0 | no bets | no bets | $100 |

* **No pre-game rule has ever cleared the fund's bar.** The logic would have stayed in cash for
  the whole 14 months. Chasing whatever looked positive lost 69% of the bankroll.
* **The double-double under does not survive the move to pre-game.** Bought before the game at
  35-80c: 635 bets, won 63.1%, needed 62.9% with the fee, +0.4% (z 0.1). In-game the same bet
  returned +15.6% after activation. The edge was in reacting to the game, not in the pick.
* Pre-game as a whole, after fees: unders -1.7% last season and -2.3% this season; overs -11.1%
  and -6.2%. Unders remain the less bad side, which is why lottery cards are built from them.
* Only 2 of 103 pre-game rules with 150+ bets in each season were positive in both: NFL receiving
  yards unders at 35-50c (+3.8%, +6.6%; z 0.6 and 1.3) and NBA threes unders at 35-50c (+1.3%,
  +0.8%). With 103 rules looked at, two weak positives are what chance produces. Neither is bet.

What the bot does now: the core sleeve issues a single-bet slip only when a pre-game rule is active
(3 standard errors on earlier nights and over 3% across its last 20). None is, so the core stays in
cash and says so. The check runs every day, so a rule that earns it will start getting slips. The
lottery cards were already pre-game and are unchanged; they remain negative expected value.

## 21. Post-mortem of the Nov 23 cards, and the leg logic redone (`python -m bot legs`, `python -m bot nflform`)

**What happened.** Two Monday-night test cards (all unders, one game): 2 of 5 and 7 of 9 legs hit, both
lost. Across them 9 of 14 legs hit (64%) at an average price near 63c. That night's NBA cards: 34 of 58
under legs hit (59%); none of the cards won.

**What was wrong with how the legs were chosen.** For NFL legs the builder used the price and the prop
type and nothing else. It never read a game log, did not know Jacory Croskey-Merritt was in his first
game back after six weeks, and did not know which team was the underdog. Each idea for why legs missed
was then tested on two seasons of settled Kalshi unders, using the price a taker really paid plus the fee
(11,698 NFL and 22,103 NBA unders).

| idea | test | result |
|---|---|---|
| Too many legs from one game or one team sink together | outcome correlation of over 900,000 same-game pairs of unders from different players | No. Different players' unders are close to independent, teammates included (NFL teammates' receptions +0.01). Only a quarterback's passing yards and his receivers move together (+0.21 to +0.26), and the same player's own props (+0.2 to +0.4). |
| Don't take unders against a player's recent hot games | NFL unders where his last 6 games say "over" | No. They returned -0.1%, better than unders his cold streak "supports" (-3.8%). Recent form adds nothing after the price in either season (z -0.9, -1.5). |
| Running-back reception unders are bad (2 of the misses) | by position | Not established: -1.1% last season, -8.0% this season. |
| Players returning from weeks out are unpredictable | first game back after 3+ weeks (NFL) | Leans bad in both seasons: -13.1%, -5.8% (baseline -3.7%, -1.2%). **lean avoid.** |
| Game script matters | by point spread | **Rushing unders on 3+ point underdogs: +2.4% and +14.9% after fees (967 bets, 2.9 standard errors over the baseline). PREFER.** Rushing unders on favourites: -5.5%, -11.6%. |
| (NBA) time off | first game back after 7+ days | **+8.3% and +7.1% after fees (1,206 bets, 3.1 standard errors over the baseline). PREFER.** The opposite sign to the NFL, with a reason in each case: NBA players come back on minute limits. |
| Tight-end reception unders | by position | +8.0%, +3.4%: both seasons, 1.4 standard errors. **lean prefer.** |

Every other category tested (12 NFL, 14 NBA, all written down before grading) is neutral.
With 26 categories, two clearing 2.9 and 3.1 standard errors over the baseline is more than chance
should give, but neither is proven as a bet: against zero they sit at 2.4 and 2.2.

**What changed.**
* Every NFL leg now carries facts: his last six games against the line, weeks since he last played,
  position, and whether Kalshi's own moneyline makes his team an underdog. Printed on the slip.
* Cards drop avoid and lean-avoid legs and take prefer legs first (`bot/legs.py`). Categories are
  re-graded daily, so a label that stops holding in both seasons falls back to neutral.
* All 26 categories went into the fund's pre-game walk-forward test as candidate rules. None is at
  3 standard errors, so the core sleeve is still in cash. The fund report now prints a watch list:
  underdog rushing unders (2.4) and NBA first-game-back unders (2.2) lead it. If either reaches
  the bar it starts getting single-bet slips without further changes.

**What did not change, and should not be expected to.** Re-dealt with the new logic, the same Monday
game still loses both cards (3 of 6 and 7 of 9): Washington was the underdog and won, so its rushers
went over, the opposite of the usual script. The only leg the new logic would have saved is the
returning player. One game cannot test any of this. The cards remain negative expected value; the
changes move a few legs per card toward kinds of under that have paid better.

## 22. Hedging, arbitrage and what outside research says (`python -m bot hedge`, `python -m bot arb`)

**Hedging on Kalshi.** A filled combo cannot be cancelled or reversed, and selling it back needs a
market maker to quote, with no guarantee (Kalshi help, "Combos"). The practical hedge is the opposite
contract of the next leg in its own market, bought before that game starts, which fits pre-game-only
betting: early games finish, the card is alive, the late game has not begun. For a ticket worth W if
the next leg hits, an opposite contract costing c with the fee, the leg's fair chance p and bankroll B:

* lock: buy W contracts; the same amount is banked either way, W(1 - c) less the stake.
* break-even: buy stake / (1 - c); the stake comes back if the leg misses.
* Kelly: h = (1 - p)(B + W)/c - pB/(1 - c), between 0 and W. At a fair price this is the full lock;
  spread and the second fee shrink it, and for a ticket small next to the bankroll it is zero.

A hedge never adds expected value. Example at Kalshi-like prices ($5 ticket worth $250 if the last
leg hits, 60% leg, other side 42c, $100 bankroll): locking banks $135.74 either way and gives up
$9.26 of expected value; the Kelly hedge is 214 of the 250 contracts. With a $2,500 ticket the same
sum says hedge almost all of it; with a $10 ticket it says do not hedge.

**Arbitrage inside Kalshi.** `python -m bot arb` checks two-sided games (YES on both teams for under
$1) and every player ladder (YES on a lower line plus NO on a higher line always pays $1, and $2 in
between: a middle). On 2026-11-24: 5 games, 211 ladders, 1,044 pairs, no arbitrage. A certain $1 from
both teams costs $1.034 to $1.043. The average middle costs 5.0 cents more than it is worth and none
of the 1,044 had a positive expected value: two spreads and two fees eat it.

**Outside evidence (found by search; the papers' own sites were not reachable from here, so these
are as reported in abstracts and summaries).**
* Kalshi takers and long shots: Buergi, Deng and Whelan (2025) find the favourite-longshot bias on
  Kalshi, stronger for takers. Bartlett and O'Hara (2026, 41.6M trades) find traders overbuy YES in
  markets that mostly settle NO. Both match what this repo measured: overs and long shots lose most.
* Parlays: Nevada books have held about 31% on parlays against 5.6% on other bets since 1984 (UNLV
  gaming research). Press reports put Kalshi combo takers' losses at about $1.0B on $7.1B staked,
  14 cents per dollar; this repo's own sample so far says 28 cents, give or take 14.
* Streaks: Paul, Weinbach and Humphreys (2011) find bettors chase streaks but fading them is not
  profitable, the same as section 21's result on recent form.
* Hedging: Whelan (2025) shows optimal betting can include hedges, but a bookmaker's margin makes
  the aggressive version lose more, the same conclusion as the Kelly line above.
* An independent NFL study (MaddocksJ27/NFL-Market-Efficiency) reports prop unders hitting 52-54%
  and concludes the gain is from shopping lines across books, not mispricing. On one venue that
  advantage does not exist.

**Kalshi mechanics worth knowing.** A player who does not play does not void a combo: the leg settles
at its fair price and the combo pays the product. "Protected" player markets (YES only, pricier)
settle at the pre-game price if the player leaves after half-time.

**Tools reviewed and not adopted.** No hedge or arbitrage library is worth a dependency (the sums are
a few lines). georgedouzas/sports-betting is soccer; odds scrapers and The Odds API serve sportsbooks,
not a Kalshi-only bot; pykalshi duplicates the client here. Possible next data: nflverse snap counts,
injuries and depth charts (reachable, free) for "a starter is out" categories, and a point-in-time
NBA injury-report archive (yangzhou-tysportsanalytics, unreviewed).

## 23. QuantSport: team rating models against the price, and what the card budgets do (`python -m quantsport research | backtest`)

**Team rating models (NBA, NFL, college football).** A points-based rating per team, updated after every game,
with home advantage and season carry-over tuned on earlier seasons only (`quantsport/models/ratings.py`). Out of
sample it picks 68% of NBA winners, 64% of NFL and 74% of college football, with margin errors of 11.3, 10.0 and
14.2 points. Then the only question that matters: does it add anything to Kalshi's pre-game price?

57,516 settled Kalshi game markets (winner, spread, total, team total; NBA and NFL from two seasons of fills,
college football from this season's price history, 1,180 games). Fitted on the earlier 60% of dates, scored on
the later 40%:

| sport | market | markets | model log-loss | price log-loss | model weight after the price (z) | blend beats price later? |
|---|---|---|---|---|---|---|
| NBA | winner | 2,947 | 0.601 | 0.573 | +0.08 (0.4) | no |
| NBA | spread | 18,721 | 0.573 | 0.550 | +0.03 (0.2) | no |
| NBA | total | 17,714 | 0.604 | 0.586 | +0.06 (0.4) | no |
| NBA | team total | 3,514 | 0.599 | 0.574 | -0.19 (-0.7) | no |
| NFL | winner | 762 | 0.627 | 0.598 | -0.51 (-1.1) | no |
| NFL | spread | 6,603 | 0.567 | 0.540 | -0.36 (-1.5) | no |
| NFL | total | 4,987 | 0.597 | 0.583 | +0.28 (1.0) | no |
| NFL | team total | 1,203 | 0.558 | 0.530 | -0.47 (-1.1) | no |
| NCAAF | winner | 1,065 | 0.590 | 0.547 | -0.36 (-1.4) | no |

The price is better in all nine, and the model's weight after the price is never distinguishable from zero.

**Negative control.** Had the fund trusted the model and bet every disagreement of 5+ points (no spread
charged, which flatters it): 23,666 bets, -6.8% per dollar; negative in all nine groups.

**So the fund's estimate rule is:** our probability is Kalshi's price unless a blend has beaten the price on
later data with a model weight over 2 standard errors. Under that rule the chain would have accepted zero singles
from game models in the whole history, and on the live slate of 2026-11-25 (19,870 buyable sides, 111 games) it
returned NO BET on everything. The best expected return available anywhere was -0.6% per dollar.

**Card budgets.** Monte Carlo of the card tiers (4,000 paths):
* $17 of cards a night from a $100 bankroll: broke in 81-97% of paths within 100 nights, even at fair quotes.
* $10 a week with a 30% drawdown stop: stopped out in 81-97% of paths within 26 weeks.
* The design adopted: each card tier has its own pot (10% / 5% / 5% of bankroll) and a weekly pace (3 / 2 / 1
  cards at $1). The cards can then never cost more than $20 of $100. At the 72-86% of fair value real combo
  buyers have been getting, the degen pot is empty within half a year in 95% of paths and the median bankroll ends
  at $81; at fair quotes the median is $93.

**Rule change.** The fund's cards now take one leg per game and never put a game on two cards (the owner's
QuantSport specification). The earlier short-slate change that allowed five legs per game applies only to the old
`python -m bot slip` cards, which the daily run no longer builds.

## 24. Injury report, price-level error bars, the watch list's own record, college football totals (2026-11-27)

**The NFL injury report is now read** (nflverse `injuries`, the official weekly report; `bot/nfl_form.py`
`week_flags`). For each prop it gives the player's own game status and two things derived from it: whether the
lead player at his position is out or doubtful, and whether his starting quarterback is. Each was tested as a leg
category on two seasons of unders at real taker prices (`python -m bot legs`):

| category | last season | this season | bets | vs all NFL unders | label |
|---|---|---|---|---|---|
| his starting quarterback is out | +8.0% | +9.7% | 123 / 255 | +11.1 points (z 0.9) | lean prefer |
| listed questionable | | | under 100 in a season | | too few bets |
| the lead player at his position is out | | | under 100 in a season | | too few bets |

"Quarterback out" points the right way in both seasons but is 378 bets from a few dozen games, well short of
significant. It is a lean, not a rule, and it moves no estimate. What the report does change today: the thesis on
an NFL prop now states the player's status, or says plainly that his team has not posted game statuses yet
(they come out about two days before the game). Game-day inactives are still not verified.

**A category's edge is now measured at the price level it is applied to.** The watch list used one hit-minus-price
gap per category for every line, which overstated cheap lines. The gap is now measured in three price bands
(20-45c, 45-65c, 65-85c), pulled toward the overall gap, with a wider error bar than the category as a whole
(a band has a third of the bets), and is not applied at all outside 20-85c:

| category | overall gap | 20-45c | 45-65c | 65-85c |
|---|---|---|---|---|
| NFL rushing unders on 3+ point underdogs | 5.5c +/- 2.3 | 4.6 +/- 2.9 | 6.9 +/- 2.8 | 4.7 +/- 3.0 |
| NBA first game back after 7+ days off | 5.7c +/- 1.9 | 6.8 +/- 2.5 | 5.5 +/- 2.4 | 4.6 +/- 2.6 |

With the honest error bars neither clears the 3-standard-error bar for a single bet at any price level, so both
stay on the watch list (WAIT), as before. A player's ladder of lines (40+, 50+, 60+ yards) now counts as one
position, not eight.

**The watch list now keeps its own record** (`python -m quantsport performance`). Every WAIT is graded as if it
had been bought at the ask it saw, one position per player, prop type and day, with the error bar counted by game.
The price just before the game is stored too (closing-line value). First read, two NFL games: 6 positions, 5 won,
needing 53%. That is two games: it says nothing yet, and the report says so until there are 200 positions over
40 games.

**College football totals: the price wins again.** 16,216 settled total markets with a pre-game price were
downloaded; 8,336 on 446 games match a rated game. Rating model log-loss 0.587, Kalshi's price 0.542. In the
blend the model's weight is negative (z -3.1): where the model disagreed with the price, the price was right.
The blend did not beat the price on later dates (0.5296 vs 0.5291), so the estimate for college totals stays
Kalshi's price.

**College football spreads: same answer.** 22,684 settled spread markets with a pre-game price; 12,310 on 446
games match a rated game. Model log-loss 0.568, price 0.532, model weight after the price -0.15 (z -1.0), blend
no better than the price on later dates (0.5441 vs 0.5433). That is 11 of 11 sport-and-market groups where the
rating model adds nothing to Kalshi's price.

## 25. Resting orders: does bidding under the ask and waiting beat paying it? (`python -m bot resting`, 2026-11-27)

The watch list is made of bets that are positive at our estimate but not after the taker fee and the spread. A
resting order on a player prop pays no fee (prop fee type `quadratic`: takers only) and buys below the ask, so
the obvious question is whether waiting turns those into bets. Tested on every real pre-game prop fill on file
(3.16M fills, both seasons): for 119,000 market-sides the first ask a taker paid between 3 hours and 30 minutes
before the start is "the ask we saw"; we bid 1-5c under it; the bid counts as filled only if a later pre-game
trade printed below it. Filled bids are graded at the bid with no fee.

| group | 2c under: filled | return on filled | paying the ask at once | won minus ask: filled / not filled |
|---|---|---|---|---|
| NBA all prop unders (22,906 orders) | 34% | -2.0% +/- 1.3 | -2.3% +/- 0.8 | -3.0 / +2.0 points |
| NFL all prop unders (11,677) | 32% | -0.2% +/- 2.6 | -0.7% +/- 1.5 | -2.1 / +2.6 |
| NBA all prop overs (57,700) | 14% | -3.1% +/- 1.3 | -7.5% +/- 0.7 | -3.6 / -2.1 |
| NFL all prop overs (26,328) | 16% | -4.0% +/- 2.7 | -9.0% +/- 1.4 | -3.9 / -2.7 |
| NBA first game back after 7+ days (932) | 34% | +5.6% +/- 4.9 | +9.5% +/- 4.0 | +0.8 / +9.9 |
| NFL rushing unders, 3+ point underdogs (801) | 34% | +8.6% +/- 7.2 | +5.3% +/- 4.7 | +2.5 / +5.5 |

* **Adverse selection is real and about the size of the saving.** On NBA unders a bid 2c under the ask is filled
  a third of the time, and the filled ones won 3 points less often than the ask implied while the unfilled ones
  won 2 points more. Your bid gets hit when someone wants out, which is more often when the bet is about to lose.
  The 2c and the fee you save are spent on that: -2.0% resting against -2.3% paying up.
* **It does not rescue the watch list.** For the NBA first-game-back unders the filled bids did worse than paying
  the ask (+5.6% vs +9.5%): the good ones are the ones nobody sold. For NFL underdog rushing unders the filled
  bids did a little better (+8.6% vs +5.3%) with an error bar of 7. Season by season no group is over 1.2 standard
  errors. No rule comes out of this.
* Resting bids on overs lose about half as much as paying the ask, and still lose.
* Bids deeper under the ask fill less (5c under: 11-14%) and are selected harder.

What changed in the system: a wait-listed bet now quotes this ("a resting bid 2c under was filled 34% of the
time; the filled ones returned ... against ... for paying the ask"), so the wait price is read as "buy if the
ask comes down to here", not "leave a bid here and expect the same edge".

**Open-position monitor** (`python -m quantsport positions`): each open card with legs won, legs to play at
their current prices, the chance it still cashes and what it is worth. A combo cannot be sold back, so when one
leg is left it prints the hedge three ways (break-even, Kelly, full lock) and what each gives up in expected value.

## Bottom line
* **One confirmed rule:** the in-game NBA double-double under. A walk-forward backtest that never
  sees the future activated it on 2026-04-01 and it has returned +15.6% on 605 bets since. It is
  small-capacity (about $5 per market) and is being paper-traded live with the fund's core sleeve.
* **The logic as a whole** made +9.7% on stakes last season and -1.1% this season: most of what it
  found last season has stopped working, so past results are not a promise.
* **Do not buy prop overs in-game.** They lost in both seasons, both sports and every price band.
* The in-game moneyline rule that looked best this season lost on 1,268 archived games and is
  graded "failed". Every rule is graded on games it was not found on.
* Matchup and style factors (form, defense-vs-role, pace, rest, scheme proxies) add nothing to
  Kalshi's price on 68,262 props; they are printed as context for each leg, not used as an edge.
* Kalshi's price is a better "will it hit" predictor than any model or simulation tested; pre-game,
  buyers lose a little in nearly every band and long shots under 20c lose the most.
* Resting orders (bid under the ask and wait) do not beat paying the ask on props: the bids that fill
  are the ones that go on to lose more often. Measured on 3.16M pre-game fills (section 25).
* Combos are at best a fair bet: 3-8 legs priced 35-80c on tight books, bought within ~10% of the
  legs' mid value. Big lottery combos sold at about 15% of fair value in the one large example.
* Kalshi accepts all-UNDER prop cards as one combo. Same-game legs hit together more often than
  their prices multiplied (1.0x at one leg per game, 1.7x at five or more), so a slip's odds are
  only as good as its legs are unrelated. What buyers paid for real combos is still being measured.
* Player type against defensive style is real in the box score for some pairings (spot-up shooters,
  rim finishers, playmakers) and absent or reversed for others (pull-up shooters against drop
  coverage score less; receiver man/zone splits are mostly noise). Kalshi's price already contains
  the real ones.
* Pre-game only (the owner's choice): no rule clears the bar on 188,029 pre-game bets, so the core
  sleeve is in cash. The double-double under that worked in-game is break-even before the game.
* Same-game legs from different players are close to independent; recent form does not help after
  the price. Two kinds of pre-game under have paid in both seasons and are on the watch list, not yet
  bet: NFL rushing unders on 3+ point underdogs, and NBA unders in a player's first game back.
* Team rating models for NBA, NFL and college football lose to Kalshi's pre-game price in all nine
  sport-by-market tests (57,516 markets); betting their disagreements returned -6.8%. The fund's
  probability is therefore the price, and the decision chain's normal answer is NO BET.
* Lottery cards are a cost, not a strategy: they now come out of small fixed pots, so they can
  never cost more than a fifth of the bankroll.
* No real money has been placed.
