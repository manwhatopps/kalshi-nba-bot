"""Tonight's NBA injury report, read fresh, so an NBA prop is never judged without knowing who is out.

Source: ESPN's public injury feed (one call, every team). Each listed player gets one plain tag:
    out | doubtful | questionable | probable | day-to-day
taken from the report's own wording first ("questionable for Friday's game") and its status fields second.

What it is used for:
  * the thesis of every NBA prop states the player's status, or that he is not listed
  * a prop on a player listed out, doubtful or questionable is never accepted and never goes on a card
  * rotation teammates who are out are stated as context (more of the work falls to those who play)

What it is NOT used for: moving an estimate. There is no history of this report on file, so "questionable" and
"teammate out" cannot be tested against Kalshi's price yet. Every read is appended to data/nba_injury_flags.parquet
(date, player, tag) so they can be graded as leg categories once enough nights have settled.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from . import config
from .data import norm_name

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CACHE = config.DATA_DIR / "ext" / "nba" / "injuries_latest.json"
FLAGS = config.DATA_DIR / "nba_injury_flags.parquet"
UNAVAILABLE = ("out", "doubtful", "questionable")            # availability not confirmed: no accept, no card leg
WORDS = (("questionable", "questionable"), ("doubtful", "doubtful"), ("probable", "probable"),
         ("game-time decision", "questionable"), ("game time decision", "questionable"))
OUT_WORDS = re.compile(r"\b(is out|ruled out|will not play|won't play|won't suit up|will miss|out for|out indefinitely|remains out|sidelined)\b")


def tag_of(status: str, fantasy: str, comment: str) -> str:
    """One plain tag from the report's wording, then its status fields."""
    c = (comment or "").lower()
    for word, tag in WORDS:
        if word in c:
            return tag
    if OUT_WORDS.search(c):
        return "out"
    f, s = (fantasy or "").upper(), (status or "").lower()
    if f == "OUT" or s == "out":
        return "out"
    if f == "GTD" or "day" in s:
        return "day-to-day"
    return s or "listed"


def parse(raw: dict) -> pd.DataFrame:
    rows = []
    for team in raw.get("injuries") or []:
        for i in team.get("injuries") or []:
            a, d = i.get("athlete") or {}, i.get("details") or {}
            name = a.get("displayName") or ""
            if not name:
                continue
            rows.append(dict(key=norm_name(name), player=name, espn_team=team.get("displayName", ""),
                             tag=tag_of(i.get("status", ""), (d.get("fantasyStatus") or {}).get("abbreviation", ""), i.get("shortComment", "")),
                             injury=" ".join(x for x in (d.get("side"), d.get("type"), d.get("detail")) if x and x != "Not Specified").lower(),
                             comment=i.get("shortComment") or "", updated=i.get("date") or "", back=d.get("returnDate") or ""))
    d = pd.DataFrame(rows, columns=["key", "player", "espn_team", "tag", "injury", "comment", "updated", "back"])
    return d.sort_values("updated").drop_duplicates("key", keep="last").reset_index(drop=True)


def fetch(max_age_min: float = 20, log: bool = True) -> pd.DataFrame:
    """The current report. Cached for a few minutes; empty (with attrs['error']) if the source cannot be reached."""
    raw = None
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < max_age_min * 60:
        raw = json.loads(CACHE.read_text())
    else:
        try:
            r = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            raw = r.json()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(raw))
        except Exception as e:
            d = parse({})
            d.attrs["error"] = str(e)[:120]
            return d
    d = parse(raw)
    d.attrs["read"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes")
    if log and len(d):
        _log(d)
    return d


def _log(d: pd.DataFrame) -> None:
    """One row per listed player per day (the latest read of the day wins): the history to grade later."""
    day = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    new = d[["key", "player", "tag", "injury"]].assign(date=day)
    old = pd.read_parquet(FLAGS) if FLAGS.exists() else new.iloc[:0]
    pd.concat([old[old.date != day], new], ignore_index=True).to_parquet(FLAGS, index=False)


def when(updated: str) -> str:
    try:
        return pd.Timestamp(updated).tz_convert("America/New_York").strftime("%b %-d %-I:%M%p ET").replace("AM", "am").replace("PM", "pm")
    except Exception:
        return "time not given"


def status_line(row) -> str:
    """The thesis line for a listed player. Starts 'injury report: listed OUT ...' so downstream checks can read it."""
    what = f" ({row.injury})" if row.injury else ""
    return f"injury report: listed {row.tag.upper()}{what}, updated {when(row.updated)}"


def blocks(why: list[str] | None) -> str:
    """'out' / 'doubtful' / 'questionable' if the leg's own player is listed that way, else ''. Works for NBA and NFL lines."""
    for w in why or []:
        m = re.match(r"injury report: listed ([A-Z\-]+)", str(w))
        if m and m.group(1).lower() in UNAVAILABLE:
            return m.group(1).lower()
    return ""


def run(args=None) -> None:
    d = fetch(max_age_min=0)
    if d.attrs.get("error"):
        print(f"NBA injury report unavailable: {d.attrs['error']}")
        return
    print(f"\nNBA INJURY REPORT (ESPN, read {d.attrs['read']}): {len(d)} players listed")
    print(d.tag.value_counts().to_string())
    pd.set_option("display.width", 220)
    show = d[d.tag.isin(["questionable", "doubtful", "probable", "day-to-day"])].assign(updated=lambda x: x.updated.map(when))
    print("\nNot ruled out (the ones that move tonight's props):")
    print(show[["player", "espn_team", "tag", "injury", "updated"]].to_string(index=False))
