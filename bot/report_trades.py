"""`python -m bot trades`: what takers and resting orders actually earned, fill by fill, and whether
a rule you could follow (buy once at first in-game touch of a price) made money."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import trades as T
from .favorites import slate_date
from .kalshi import data_client

pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 300)
DISCOVERED_THROUGH = "2026-11-21"
PROPS = ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA", "KXNBASTL", "KXNBABLK", "KXNBASTOCK",
         "KXNBA2D", "KXNBA3D", "KXNFLTD", "KXNFLREC", "KXNFLRSH", "KXNFLPASS", "KXNFLFIRSTTD", "KXNFLRR",
         "KXNFLANYTD", "KXNFL2TD")


def group(s: str) -> str:
    if s in ("KXNBAGAME", "KXNFLGAME"):
        return "1 moneyline"
    if "SPREAD" in s:
        return "2 spreads"
    if "TOTAL" in s:
        return "3 totals"
    return "4 player props" if s.startswith(PROPS) else "5 other (halves, quarters...)"


def _fmt(tab, pct=("roi", "se")):
    tab = tab.copy()
    for c in pct:
        tab[c] = tab[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
    return tab.round(3).to_string(index=False)


def refresh(since: date):
    T.download(since)
    n = pd.concat([pd.read_parquet(f, columns=["ticker"]).ticker.str.split("-").str[0].value_counts()
                   for f in sorted(T.TRADES_DIR.glob("2*.parquet"))[-3:]], axis=1).sum(1)
    T.fetch_results(data_client(), [s for s in n.index if n[s] >= 3000],
                    datetime(since.year, since.month, since.day, tzinfo=timezone.utc))
    T.aggregate()
    T.first_touch()
    from . import props_study
    props_study.first_in_band(T.TRADES_DIR, pd.read_parquet(T.RESULTS), T.config.DATA_DIR / "prop_touch.parquet")


def run(args):
    if args.download:
        y, m, d = map(int, args.since.split("-"))
        refresh(date(y, m, d))
    a = T.with_roles(T.load_agg())
    a = a[a.league == args.league]
    ph = T.PHASES[:3]
    print(f"\n{args.league}: {int(a.n.sum()):,} fills, {a.series.nunique()} market types, {a.game.nunique()} games. "
          "ROI after Kalshi fees, every fill counted once; se clustered by game.")
    for role, title in (("taker", "TAKERS (bought at the quoted price)"), ("maker", "MAKERS (resting order got filled)")):
        tab = T.role_table(a, role, "n")
        tab = tab[tab.phase.isin(ph)]
        tab["phase"] = pd.Categorical(tab.phase, ph, ordered=True)
        print(f"\n{title} - by the price they paid")
        print(_fmt(tab.sort_values(["phase", "avg_price"])))
    fp = T.config.DATA_DIR / "first_touch.parquet"
    ft = pd.read_parquet(fp) if fp.exists() else T.first_touch()
    ft = ft[ft.league == args.league].assign(group=lambda d: d.series.map(group))
    print("\nA RULE YOU CAN FOLLOW: buy one contract the first time either side can be bought at >= threshold in-game")
    print(_fmt(T.touch_table(ft, ["threshold", "group"])))
    ml = ft[ft.group == "1 moneyline"].assign(date=lambda d: d.game.map(slate_date))
    if len(ml):
        nights = sorted(ml.date.dropna().unique())
        ml = ml.assign(half=np.where(ml.date < nights[len(nights) // 2], "first half of season", "second half"))
        print("\nMoneyline only, split by half of the season (each game appears twice: YES one team / NO the other):")
        print(_fmt(T.touch_table(ml, ["threshold", "half"])))
        # The rule was found on games through 2026-11-21. Everything after is a genuine out-of-sample test.
        new = ml[ml.date > pd.Timestamp(DISCOVERED_THROUGH)]
        print(f"\nOUT-OF-SAMPLE (games after {DISCOVERED_THROUGH}, i.e. after the rule was found):")
        print(_fmt(T.touch_table(new, ["threshold"])) if len(new) else "  no games yet - this fills in as new days of trades arrive")
    return a, ft
