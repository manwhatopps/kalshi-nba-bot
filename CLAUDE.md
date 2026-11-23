# Kalshi sports bot — working notes

Research + paper-trading bot for Kalshi NBA/NFL markets. Read `FINDINGS.md` before changing strategy code:
it records what the backtests on Kalshi's own settled markets and 45M real fills actually show.

## Rules
- **Paper only.** Never place real orders (`place --live`) unless the owner explicitly says to in the
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
`python -m pytest -q tests` (6 tests, ~2 s).
