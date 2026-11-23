#!/usr/bin/env bash
# Resumable backfill of Kalshi settled markets (rate-limited: ~25 min per NBA market type).
cd "$(dirname "$0")/.." && . .venv/bin/activate
python -c "
from bot.history import download
download(days=${1:-35}, leagues=('NBA',))
download(days=${1:-35}, leagues=('NFL',))
" >> data/history_download.log 2>&1
