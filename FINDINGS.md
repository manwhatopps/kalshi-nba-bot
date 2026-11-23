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

What does hold pre-game: **unders as a group beat their price by 2.8 points last season and 0.9 this
season** - the structural over/under tilt, not matchup insight.

So the chain is context, not an edge. Degen cards print it for every leg (`--why`): recent form vs
the line, minutes trend, role and what this defense gives that role, pace, opponent defense, rest.
It is used only to drop legs where it clearly points to the over and to order otherwise-equal legs.

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
* Combos are at best a fair bet: 3-8 legs priced 35-80c on tight books, bought within ~10% of the
  legs' mid value. Big lottery combos sold at about 15% of fair value in the one large example.
* No real money has been placed.
