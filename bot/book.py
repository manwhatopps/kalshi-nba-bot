"""Real-time prices from Kalshi's order book (public, no key): what a slip will really cost.

`GET /markets/orderbooks` returns, for up to 100 markets per call, every resting YES bid and NO bid with
its size. There are no asks in a binary book: buying NO at price p means matching a YES bid at 1 - p.
So the true cost of N contracts is found by walking the other side's bids from the best down, which the
one-line quote on `/markets` (best bid/ask and its size only) cannot tell you.
"""
from __future__ import annotations

import math

from .kalshi import Kalshi

TAKER = 0.07      # 'quadratic' fee type, multiplier 1: fee = 0.07 * contracts * price * (1 - price), rounded up to the cent per order


def orderbooks(k: Kalshi, tickers: list[str]) -> dict:
    """{ticker: {'yes': [(price, size), ...], 'no': [...]}} with bids sorted best (highest) first."""
    out = {}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), 100):
        d = k.request("GET", "/markets/orderbooks", params={"tickers": tickers[i:i + 100]})
        for ob in d.get("orderbooks") or []:
            fp = ob.get("orderbook_fp") or {}
            out[ob["ticker"]] = {s: sorted(((float(p), float(q)) for p, q in fp.get(f"{s}_dollars") or []), reverse=True)
                                 for s in ("yes", "no")}
    return out


def offers(book: dict, side: str) -> list[tuple[float, float]]:
    """What can be bought on `side`, cheapest first: (price, size)."""
    other = book.get("no" if side == "yes" else "yes") or []
    return [(round(1 - p, 4), q) for p, q in other]


def top(book: dict) -> dict:
    """Best bid, ask and mid for YES and NO from the book."""
    yb = book["yes"][0][0] if book.get("yes") else 0.0
    nb = book["no"][0][0] if book.get("no") else 0.0
    ya, na = (round(1 - nb, 4) if nb else 1.0), (round(1 - yb, 4) if yb else 1.0)
    return dict(yes_bid=yb, yes_ask=ya, no_bid=nb, no_ask=na, spread=round(ya - yb, 4),
                yes_mid=(yb + ya) / 2, no_mid=(nb + na) / 2)


def fee_dollars(fills: list[tuple[float, float]]) -> float:
    """Taker fee for one order filled at these (price, contracts): summed, then rounded up to the cent."""
    raw = sum(TAKER * n * p * (1 - p) for p, n in fills)
    return math.ceil(raw * 100 - 1e-9) / 100 if raw > 0 else 0.0


def buy(book: dict, side: str, stake: float, limit: float, whole: bool = True) -> dict:
    """Spend up to `stake` (fee included) buying `side`, never paying more than `limit` per contract.
    Returns contracts, cost with fee, average and worst price paid, and how much was on offer within the limit."""
    fills, spent, avail = [], 0.0, 0.0
    for p, q in offers(book, side):
        if p > limit + 1e-9:
            break
        avail += q
        n = q if not whole else math.floor(q)
        while n > 0:      # the most of this level the remaining stake affords, fee included
            trial = fills + [(p, n)]
            if sum(a * b for a, b in trial) + fee_dollars(trial) <= stake + 1e-9:
                break
            n -= 1
        if n > 0:
            fills.append((p, float(n)))
        if n < (q if not whole else math.floor(q)):
            break
    c = sum(n for _, n in fills)
    principal = sum(p * n for p, n in fills)
    return dict(contracts=c, cost=round(principal + fee_dollars(fills), 2), fee=fee_dollars(fills),
                avg=principal / c if c else float("nan"), worst=fills[-1][0] if fills else float("nan"),
                levels=len(fills), offered_within_limit=avail, fills=fills)


def max_price_for_payout(mult: float) -> float:
    """Highest contract price at which $1 (fee included) still returns `mult` dollars: p + fee(p) = 1 / mult."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid + TAKER * mid * (1 - mid) > 1 / mult:
            hi = mid
        else:
            lo = mid
    return lo
