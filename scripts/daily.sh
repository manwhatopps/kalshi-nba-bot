#!/usr/bin/env bash
# Daily paper-trading run: top up Kalshi history, grade yesterday, re-test the strategies on all
# history, and log tonight's cards as paper trades. Results accumulate in data/ and reports/.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p reports
export BOT_LEDGER_DIR=data/ledger
out="reports/$(date -u +%F).md"
python -m bot history --days 4 > /dev/null 2>&1 || true     # top up settled markets (rate-limited)
{
  echo "# Kalshi sports bot — $(date -u '+%F %H:%M UTC')"
  echo; echo '## Yesterday graded'; echo '```'
  python -m bot settle 2>&1 | tail -25
  echo '```'; echo; echo '## Do the models beat Kalshi? (all settled props)'; echo '```'
  python -m bot models 2>&1 | tail -16
  echo '```'; echo; echo '## Favorites strategy on all history'; echo '```'
  python -m bot favorites --samples 100 2>&1 | tail -70
XX
  python -m bot card --legs 10 --min-price 0.90 2>&1 | tail -18
  python -m bot card --legs 20 --min-price 0.90 2>&1 | tail -5
  python -m bot card --legs 50 --min-price 0.90 2>&1 | tail -5
  echo '```'; echo; echo "## Tonight's value combos (paper)"; echo '```'
  python -m bot scan --hours 14 --top 6 --show-legs 8 2>&1 | sed -n '/open legs/,$p'
  echo '```'
} > "$out"
cat "$out"
