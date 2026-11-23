#!/usr/bin/env bash
# Daily paper-trading run: grade yesterday, scan today's slate, save a dated report.
# Ledgers live in data/ledger/ and are committed so results accumulate across runs.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p reports
export BOT_LEDGER_DIR=data/ledger
out="reports/$(date -u +%F).md"
{
  echo "# Kalshi NBA bot — $(date -u '+%F %H:%M UTC')"
  echo; echo '## Yesterday graded'; echo '```'
  python -m bot settle 2>&1 | tail -25
  echo '```'; echo; echo "## Tonight's multi-game combos (paper)"; echo '```'
  python -m bot scan --hours 14 --top 8 --show-legs 12 2>&1 | sed -n '/open legs/,$p'
  echo '```'; echo; echo "## Tonight's same-game parlays (paper)"; echo '```'
  python -m bot scan --hours 14 --same-game --top 5 --show-legs 0 2>&1 | sed -n '/TOP COMBOS/,$p'
  echo '```'
} > "$out"
cat "$out"
