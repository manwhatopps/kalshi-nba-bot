# Kalshi sports bot — working notes

Research + paper-trading bot for Kalshi NBA/NFL markets. Read `FINDINGS.md` before changing strategy code:
it records what the backtests on Kalshi's own settled markets and 45M real fills actually show.

## What this is
A fund, not a pick list: one bankroll split into a CORE sleeve (single bets on data-graded edges, Kelly-sized
from the low end of the edge's error range until confirmed), a small fixed DEGEN sleeve (lottery combos), and
a RESERVE. `python -m bot fund` prints the plan; settings live in `data/fund.json`. Edges are graded A/B/C in
`bot/fund.py` from `data/first_touch.parquet`; only A and B get money. A rule is graded on games it was NOT
found on (the prior-season archive plus games since discovery). As of 2026-11-23 the moneyline rules grade
C (failed on the archive) and one prop rule grades B: the in-game NBA double-double UNDER at 50-80c
(`fund.dd_under_bets`, from `data/prop_touch*.parquet`). Props matter as much as game lines: test both.
`python -m bot walk` (`bot/walkforward.py`) is the honest test of the whole logic: each night it sees only
earlier nights, activates rules whose record clears 3 standard errors plus a 20-night recency check, and
bets them. Any change to the logic must be judged there, season by season, not on a hand-picked rule. `python -m bot sim` replays real nights in
`bot/sim_env.py` (gymnasium `FundEnv`) to compare staking policies; any new rule must survive it.

`python -m bot slip` turns the plan into placeable slips (exact market, side, contracts, limit, walk-away
payout, Kalshi combo-rule check); `--quote`/`--live` need the owner's key and obey the stake caps.

`python -m bot scenarios` tests player type against defensive style (real in the box score? priced by Kalshi?)
and writes `data/scenario_table.json`, which the leg explanations quote. New scenario ideas go there first.

`python -m bot legs` grades pre-written leg categories (underdog, first game back, position, form...) on two
seasons of unders at real taker prices and writes `data/leg_categories.json`; cards drop avoid legs and take
prefer legs first, and every category is also a candidate rule in `walk --pre`. A new idea about why legs hit
or miss goes in `bot/legs.py` as a category, not into the card builder as a hunch.

**Pre-game only (owner's decision 2026-11-23):** every bet is placed before the game. No live watcher, no
in-game rules; `pre_game_only` in `data/fund.json`. The in-game double-double under (grade B) is therefore not
bet. `python -m bot walk --pre` is the test that matters now; as of 2026-11-23 no pre-game rule clears it, so
the core sleeve is in cash. Do not restart `python -m bot watch` unless the owner asks for live betting.

## Rules
- **Paper only.** Never place real orders (`place --live`, `slip --quote --live`) unless the owner explicitly says to in the
  current conversation, with a stake limit. Defaults: `KALSHI_ENV=demo`, $2 per combo, $10 per day.
- Never commit `.env`, `*.pem`, or API keys. Kalshi keys are RSA files that must stay on the owner's machine.
- Report results as measured, including negative ones. A strategy only counts if it holds out-of-sample.
- `python -m bot social` uses an undocumented Kalshi website API: manual, occasional use only.

## Daily run
`bash scripts/daily.sh` (about 20 minutes): pulls new fills from Hugging Face `TimeSeventeen/Kalshi`, fetches
results from Kalshi, grades paper trades, re-runs the studies, builds tonight's paper cards, and writes
`reports/YYYY-MM-DD.md`. Afterwards commit `data/ledger/`, `reports/`, and the summary parquet/json files
under `data/` that `.gitignore` allow-lists, then push to `main`.

Raw trade files (`data/trades/`, ~25 MB per day) are not in git. Summaries (`data/fills_agg.parquet`,
`data/first_touch.parquet`) are, and are updated incrementally — never rebuild them on a machine that
lacks the full raw history.

## Network
Needs: api.elections.kalshi.com, huggingface.co, us.aws.cdn.hf.co, cas-bridge.xethub.hf.co. stats.nba.com
blocks cloud IPs; the Hugging Face game-log dataset is the stats source.

## Tests
`python -m pytest -q tests` (14 tests, ~3 s).
