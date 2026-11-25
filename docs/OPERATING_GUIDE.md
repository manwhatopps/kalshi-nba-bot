# QuantSport operating guide

Everything runs without a Kalshi key and places nothing.

## Start
```
git clone https://github.com/manwhatopps/kalshi-nba-bot && cd kalshi-nba-bot
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q tests            # 29 tests
```

## Daily investment analysis
```
python -m quantsport settle                 # grade yesterday's paper positions and decisions
python -m quantsport daily --mode paper     # scan, estimate, chain, audit, tiers; opens paper positions; writes the dashboard
python -m quantsport performance            # P&L by tier, calibration by sport and market
```
`bash scripts/daily.sh` does all of this plus the older studies and writes `reports/YYYY-MM-DD.md`.
`--mode research` analyses without touching the paper book.

## Reviewing the investment cards
* Terminal: the report printed by `daily` (sections 3-6 are the four tiers).
* `data/quantsport/dashboard.html`: fund, scanner, each core decision with its 16-step thesis, each card with legs,
  portfolio limits, performance, and the model-versus-price table.
* A card says PENDING QUOTE: it may be bought only if Kalshi's quote pays at least the WALK-AWAY multiple. Check a
  combo you built by hand with `python -m bot combo "<link>"`; hedge a live one with `python -m bot hedge`.

## Weekly research
```
python -m quantsport research     # re-tune the rating models and re-test them against Kalshi's prices
python -m quantsport backtest     # walk-forward tables, the negative control, Monte Carlo of the card pots
python -m bot legs                # re-grade the leg categories
```
`research` needs the raw fill files for NBA/NFL prices (`python -m bot trades --download`).

## Changing risk settings
`python -m quantsport config` prints them; `python -m quantsport config --set z_edge=3 kelly_fraction=0.25` writes
`data/quantsport.json`. Card rules: `same_game_per_card` (1 = the diversification rule; raising it is the explicit
override and every such card is labelled), `reuse_game_across_cards`, `budget`, `weekly_spend`, `min_keep`.

## Paper trading
`mode: "paper"` (default). Accepted singles are filled at the real order book; cards are assumed bought at their
walk-away payout, because no public data or keyless API gives a real combo quote. Paper card results therefore test
the leg probabilities, not what a market maker would actually pay.

## Before any live order (none of this is done)
1. Kalshi account eligible in your state; read Kalshi's developer agreement.
2. API key created with only `read` and `write::trade` scopes, stored on YOUR machine; `.env` with
   `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV=prod`. Never in the repo, a chat or an env panel.
3. `python -m quantsport approve --init` at a terminal (creates the approval key outside the repo).
4. `python -m quantsport config --set mode='"live_review"'`.
5. Each order: `python -m quantsport tickets`, then `python -m quantsport approve N` and type the confirmation.
   The approval is bound to that market, side, limit, quantity and maximum cost, lasts 10 minutes, works once.
6. The sending call (`execution.place_approved`) and the combo quote path (`python -m bot slip --quote`) have never
   run against a live account. Test both on Kalshi's demo environment first.
7. A track record: the chain has accepted zero singles so far. Do not enable live trading to "see what happens".
