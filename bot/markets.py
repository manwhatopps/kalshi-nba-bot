"""Pull open Kalshi NBA markets and turn them into normalized `Leg`s."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from .kalshi import Kalshi

# series -> stat expression (sum of base stats)
PROP_SERIES = {
    "KXNBAPTS": ("pts",), "KXNBAREB": ("reb",), "KXNBAAST": ("ast",), "KXNBA3PT": ("fg3m",),
    "KXNBAPRA": ("pts", "reb", "ast"), "KXNBAPR": ("pts", "reb"), "KXNBAPA": ("pts", "ast"),
    "KXNBARA": ("reb", "ast"), "KXNBASTL": ("stl",), "KXNBABLK": ("blk",), "KXNBASTOCK": ("stl", "blk"),
}
GAME_SERIES = ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class Leg:
    ticker: str
    event_ticker: str
    series: str
    game: str            # e.g. 26NOV22OKCMIA
    away: str
    home: str
    kind: str            # prop | win | spread | total
    title: str
    yes_bid: float
    yes_ask: float
    start: datetime | None
    player: str = ""
    team: str = ""       # Kalshi team code for prop player / win / spread team
    stats: tuple = ()
    line: float = 0.0    # outcome is YES if value > line (Kalshi floor_strike)
    p_model: float | None = None
    p_book: float | None = None
    p_fair: float | None = None
    notes: list = field(default_factory=list)

    @property
    def mid(self) -> float:
        """Market-implied probability. Tight book: the mid. Wide/one-sided book: lean toward the
        ask (long shots on thin books are not as cheap as half the ask)."""
        if self.yes_bid > 0 and self.spread <= 0.04:
            return (self.yes_bid + self.yes_ask) / 2
        return self.yes_bid + 0.65 * (self.yes_ask - self.yes_bid)

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def edge(self) -> float:
        return (self.p_fair or 0) - self.yes_ask

    def label(self) -> str:
        return self.title


def _game_parts(event_ticker: str):
    g = event_ticker.split("-")[1]
    return g, g[-6:-3], g[-3:]


def parse_market(series: str, m: dict) -> Leg | None:
    if m.get("status") not in ("active", "open"):
        return None
    ask, bid = _f(m.get("yes_ask_dollars")), _f(m.get("yes_bid_dollars")) or 0.0
    if not ask or ask >= 1:
        return None
    try:
        game, away, home = _game_parts(m["event_ticker"])
    except (IndexError, KeyError):
        return None
    start = m.get("occurrence_datetime") or m.get("expected_expiration_time")
    start = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
    last = m["ticker"].split("-")[-1]
    base = dict(ticker=m["ticker"], event_ticker=m["event_ticker"], series=series, game=game, away=away,
                home=home, title=m.get("title", ""), yes_bid=bid, yes_ask=ask, start=start)
    line = _f(m.get("floor_strike"))
    if series in PROP_SERIES:
        if m.get("strike_type") != "greater" or line is None:
            return None
        seg = m["ticker"].split("-")[-2]
        return Leg(**base, kind="prop", player=m.get("title", "").split(":")[0].strip(), team=seg[:3],
                   stats=PROP_SERIES[series], line=line)
    if series == "KXNBAGAME":
        return Leg(**base, kind="win", team=last)
    if series == "KXNBASPREAD":
        if line is None:
            return None
        return Leg(**base, kind="spread", team="".join(c for c in last if c.isalpha()), line=line)
    if series == "KXNBATOTAL":
        if line is None or m.get("strike_type") != "greater":
            return None
        return Leg(**base, kind="total", line=line)
    return None


def fetch_legs(k: Kalshi, include_live: bool = False, hours_ahead: float = 36,
               series: tuple | None = None) -> list[Leg]:
    now = datetime.now(timezone.utc)
    legs = []
    for s in series or (*GAME_SERIES, *PROP_SERIES):
        try:
            ms = k.markets(s)
        except Exception as e:
            print(f"  ! {s}: {e}")
            continue
        for m in ms:
            leg = parse_market(s, m)
            if not leg or not leg.start:
                continue
            if not include_live and leg.start < now + timedelta(minutes=5):
                continue  # game started (or about to): stats model is pre-game only
            if leg.start > now + timedelta(hours=hours_ahead):
                continue
            legs.append(leg)
    return legs
