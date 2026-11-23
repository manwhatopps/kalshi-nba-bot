# Kalshi NBA Combo Bot

Scans every open Kalshi NBA market (moneyline, spread, total, and player points / rebounds /
assists / threes / PRA / steals / blocks props), prices each leg with a stats model (plus
sportsbook odds if you have a key), and builds **correlated combos** (parlays) where the fair
probability beats what Kalshi is likely to charge. Then it can request real quotes from Kalshi's
combo market makers and buy only if a quote is still +EV.

```
Kalshi markets ─┐
HF game logs ───┼─> player + team projections ─> joint game simulation (Gaussian copula)
Odds API (opt) ─┘                                      │
                       fair prob per leg = blend(model, book, Kalshi mid)
                                                       │
                    beam search over combos (1 leg per event, as Kalshi requires)
                                                       │
            ranked combos ─> RFQ to market makers ─> accept only if real quote is +EV
                                                       │
                        ledger (every leg + combo) ─> settle ─> "does the model beat Kalshi?"
```

**Read `FINDINGS.md` first: what the backtests on Kalshi's own settled markets actually show.**

## Built from existing pieces (not from scratch)

| Piece | Source |
|---|---|
| Auth signing + endpoint paths | Kalshi's official SDK `kalshi-python-sync` 3.36 / OpenAPI spec |
| Player game logs (nightly, CC0) | Hugging Face `cbratkovics/nba-game-logs` |
| Fresher stats from a home connection | `nba_api` (stats.nba.com blocks cloud servers) |
| Sportsbook de-vig | `penaltyblog` (power method) + The Odds API |
| Kalshi history for backtests | Kalshi `/markets?status=settled` + batch candlesticks |
| Extra models | scikit-learn gradient boosting + logistic stacker |
| Evidence on favorites / combo margins | Jon-Becker/prediction-market-analysis, jsteng19/kalshi-combos-research |
| Copula approach to combo pricing | idea from `pranay123-stack/kalshi-rfq-combo-pricing-engine`, priors fit on 2 seasons of real games |

## Setup (run on your own computer)

```bash
git clone <this repo> && cd kalshi-nba-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit it
python -m pytest -q tests     # 5 tests, should pass
```

**Kalshi API key:** kalshi.com → Account & Security → API Keys → *Create key*. Save the
private-key `.pem` file somewhere outside this folder and set `KALSHI_API_KEY_ID` and
`KALSHI_PRIVATE_KEY_PATH` in `.env`. The demo exchange (demo.kalshi.co) needs its **own**
account and key. Start there: `KALSHI_ENV=demo`.

Scanning needs **no key**. It reads public prod prices.

## Commands

```bash
python -m bot scan                         # value legs + top combos >= 10x
python -m bot scan --min-mult 50 --max-legs 6
python -m bot scan --same-game             # same-game parlays only (where correlation matters)
python -m bot scan --flat-stake 1          # "lottery" sizing: $1 per combo instead of Kelly
python -m bot place                        # also request REAL quotes (needs key). Buys nothing.
python -m bot place --live                 # accept quotes that are still +EV (on KALSHI_ENV)
python -m bot settle                       # grade every logged leg/combo once games finish
python -m bot report                       # model vs Kalshi Brier score, paper/real P&L
python -m bot backtest --days 21           # walk-forward calibration of the props model

# strategy research on Kalshi's own settled markets (NBA + NFL)
python -m bot history --days 35            # download settled markets + pre-game / in-game prices
python -m bot favorites                    # test "many legs at 90c+" (add --when live1 for in-game)
python -m bot models                       # score all model types vs Kalshi's price; fit live weights
python -m bot card --legs 50 --min-price 0.90   # tonight's favorites card (YES and NO legs), paper
python -m bot mixes                        # which mix of leg prices is most efficient at each payout
python -m bot mix --targets 5 10 25        # tonight's efficient mixes: few 50-72c legs (paper)
python -m bot sims                         # 20,000-simulation hit rates next to Kalshi's price
python -m bot social                       # read-only leaderboard + named trades (manual use)
bash scripts/daily.sh                      # all of the above + dated report in reports/
```

Columns: `ask/mid` = Kalshi, `model` = stats model, `book` = de-vigged sportsbook,
`fair` = blend. `est` = estimated maker quote for the combo. `corr-lift` > 1 means the legs
tend to hit together.

## How to actually use it (please read)

1. **Paper-trade for 2+ weeks.** Every `scan` logs every leg and combo. Run `settle` and
   `report` daily. The number that matters is whether `p_model` / `p_fair` gets a **lower Brier
   score than `mid`** over several hundred settled legs. If it doesn't, the model has no edge
   and no combo built from it does either.
2. Then use `place` (quotes only) for a few days. It logs real maker quotes next to the
   estimate, so you can see what Kalshi really charges.
3. Only then `place --live`, on demo first, with the stake caps in `.env`.

## Reality check

- **Parlays multiply edge *and* vig.** A 6-leg combo needs every leg's edge to be real. The
  usual result is losing most bets and, without a genuine edge, losing money overall. Big
  multipliers are long shots by definition: a fair 100x combo misses 99% of the time.
- Kalshi combo makers price correlation too. The bot's estimate assumes they do, so its EV
  comes only from where **our leg probabilities beat Kalshi's**. Only the real RFQ quote counts.
- The model is calibrated. Walk-forward over 14 game days and 15,426 player-lines, when it
  said 25% those lines hit 24.3%, and its Brier score was 0.221 vs 0.250 for a base-rate
  guess. Being calibrated is **not** the same as beating the market.
- Big model-vs-Kalshi disagreements are usually news the model can't see (injuries,
  minutes limits, rest). Legs that disagree by more than 0.20 are skipped
  (`--max-disagree`). Check injury reports before betting anyway.
- Prop books are thin. Prices jump between the scan and the bet, so `place` re-checks EV
  against the live quote.

## Safety rails

- Dry-run by default. `--live` is required to buy, and `KALSHI_ENV` defaults to `demo`.
- Stakes are 0.1× Kelly, capped per combo (`MAX_STAKE_PER_COMBO`) and per day (`MAX_DAILY_STAKE`).
- Price check: the quote's YES cost is read as the **worse** of the two possible readings of a
  maker quote (`yes_bid` vs `1 − no_bid`), and must still clear `--min-ev`.
- After a fill it checks `/portfolio/fills` and warns loudly if the fill wasn't on the YES side.
  Confirm the side semantics with one tiny trade on demo before going live.
- RFQs that aren't accepted are deleted. Kalshi charges more rate-limit tokens if you send many
  RFQs and never accept them, so `--max-rfqs` defaults to 3.

## Files

`bot/kalshi.py` API client · `bot/markets.py` market parser · `bot/data.py` game logs ·
`bot/model.py` projections + joint simulation · `bot/combos.py` pricing + search ·
`bot/odds.py` sportsbook de-vig · `bot/execute.py` RFQ execution · `bot/ledger.py` paper
trading + grading · `bot/backtest.py` calibration.
