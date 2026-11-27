#!/usr/bin/env bash
# Daily paper-trading run: pull new fills + results, grade yesterday, re-run the studies on all
# history, and log tonight's cards as paper trades. Results accumulate in data/ and reports/.
# One failing section never stops the rest: it is marked "!! SECTION FAILED" in the report and
# the script exits 1 at the end so the failure is visible.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p reports
export BOT_LEDGER_DIR=data/ledger
out="reports/$(date -u +%F).md"
fails=0

sec() {   # sec "Title" <tail-lines> command...
  local title="$1" lines="$2"; shift 2
  echo; echo "## $title"; echo '```'
  if ! "$@" > .section.out 2>&1; then
    fails=$((fails + 1)); echo "!! SECTION FAILED: $*"
  fi
  tail -n "$lines" .section.out; echo '```'
}

python -m bot trades --download --since "$(date -u -d '5 days ago' +%F)" > .refresh.out 2>&1 || { fails=$((fails + 1)); echo "refresh failed" >> .refresh.out; }
pkill -f "bot watch" 2>/dev/null   # no live watcher: every bet is placed before the game (owner's decision, 2026-11-23)

{
  echo "# Kalshi sports bot — $(date -u '+%F %H:%M UTC')"
  echo; echo "Data refresh: $(grep -c 'NBA/NFL fills' .refresh.out) new day(s) of trades added."
  sec "Yesterday graded" 25 python -m bot settle
  sec "THE FUND: tonight's allocation and what to expect" 90 python -m bot fund
  sec "SCENARIOS: player type vs defensive style, real and priced" 100 python -m bot scenarios
  sec "LEG CATEGORIES: what was knowable before the game, graded on real costs" 45 python -m bot legs
  sec "QUANTSPORT: settle paper positions" 6 python -m quantsport settle
  sec "QUANTSPORT: daily investment analysis, four tiers (paper)" 120 python -m quantsport daily --mode paper
  sec "QUANTSPORT: open positions and hedges" 40 python -m quantsport positions
  sec "QUANTSPORT: performance audit" 40 python -m quantsport performance
  sec "TEAM STYLES: all 30 teams" 34 python -m bot styles
  sec "MATCHUP FACTORS vs Kalshi's price" 22 python -m bot matchup --rebuild
  sec "PRE-GAME ONLY: walk-forward test of the fund's logic on pre-game prices" 30 python -m bot walk --pre
  sec "WALK-FORWARD BACKTEST of the fund's logic, in-game rules (reference only: not bet)" 80 python -m bot walk
  sec "SIMULATOR: double-double under rule, \$5 fillable per bet" 40 python -m bot sim --rule dd_under --max-stake 5 --episodes 150
  sec "SIMULATOR: moneyline 95c rule (failed rule, for reference)" 40 python -m bot sim --episodes 100
  sec "Every fill this season: the first-touch rule (NBA)" 60 bash -c "python -m bot trades | sed -n '/A RULE YOU CAN FOLLOW/,\$p'"
  sec "Do the models beat Kalshi? (settled props)" 16 python -m bot models
  sec "Favorites strategy on quote history" 70 python -m bot favorites --samples 100
  sec "Which mix of leg prices is most efficient" 45 python -m bot mixes --samples 60
  sec "Tonight's most probable combos by payout target (paper)" 50 python -m bot best --targets 3 5 10 25
  sec "Tonight's efficient mixes: few mid-priced legs (paper)" 45 python -m bot mix --targets 5 10 25 50
  sec "Tonight's 10x from 35-50c legs (paper)" 9 python -m bot mix --targets 10 --lo 0.35 --hi 0.50
  sec "Tonight's favorites cards (paper)" 30 bash -c "python -m bot card --legs 10 --min-price 0.90 | tail -18; python -m bot card --legs 20 --min-price 0.90 | tail -5; python -m bot card --legs 50 --min-price 0.90 | tail -5"
  sec "Tonight's value combos (paper)" 40 bash -c "python -m bot scan --hours 14 --top 6 --show-legs 8 | sed -n '/open legs/,\$p'"
  echo; echo "Sections failed: $fails"
} > "$out"
rm -f .section.out .refresh.out
cat "$out" | tail -5
[ "$fails" -eq 0 ]
