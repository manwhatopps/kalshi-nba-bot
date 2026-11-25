"""The execution boundary. This is the only code path that can send an order, and it will not without ALL of:

  * mode == live_review in data/quantsport.json and a Kalshi API key on this machine
  * an approval token issued by the owner at a terminal (`python -m quantsport approve <ticket>`), which is
      - bound to the exact market, side, limit price, quantity and maximum cost (any change invalidates it)
      - signed (HMAC-SHA256) with a key stored OUTSIDE the repository (~/.quantsport/approval.key)
      - valid for a few minutes and usable once: it is marked used before the order is sent, even if sending fails
  * a fresh quote at or under the limit, and the risk auditor's clearance at the moment of sending

Limits of this design, stated plainly: it stops accidents, stale approvals, changed orders and replays. It cannot stop
someone with full access to this machine and its key file. The real guarantee is that the Kalshi API key lives only on
the owner's computer. Sending has not been exercised against a live account: there is no key here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import store

FIELDS = ("ticker", "side", "limit_price", "quantity", "max_cost")


def key_path() -> Path:
    return Path(os.getenv("QS_APPROVAL_KEY", Path.home() / ".quantsport" / "approval.key"))


def _key(create: bool = False) -> bytes:
    p = key_path()
    if not p.exists():
        if not create:
            raise PermissionError(f"no approval key at {p}: run `python -m quantsport approve --init` at a terminal")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(32))
        p.chmod(0o600)
    return p.read_bytes()


def canonical(order: dict) -> str:
    o = {f: order[f] for f in FIELDS}
    o["limit_price"], o["quantity"], o["max_cost"] = round(float(o["limit_price"]), 4), int(o["quantity"]), round(float(o["max_cost"]), 2)
    return json.dumps(o, sort_keys=True)


def _sign(order: dict, token: str, expires: str) -> str:
    return hmac.new(_key(), f"{canonical(order)}|{token}|{expires}".encode(), hashlib.sha256).hexdigest()


def issue(con, order: dict, ttl_min: float = 10) -> str:
    """Create a one-use approval for exactly this order. Called by `approve` after the owner confirms at a terminal."""
    token, exp = uuid.uuid4().hex, (datetime.now(timezone.utc) + timedelta(minutes=ttl_min)).isoformat(timespec="seconds")
    store.insert(con, "approvals", dict(token=token, created=store.now(), expires=exp, ticker=order["ticker"], side=order["side"],
                                        limit_price=float(order["limit_price"]), quantity=int(order["quantity"]), max_cost=float(order["max_cost"]),
                                        used=None, sig=_sign(order, token, exp)))
    return token


def approve(con, order: dict, ttl_min: float = 10) -> str:
    """Interactive approval. Refuses unless a person is at a terminal and types the confirmation."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise PermissionError("approval must be given by a person at a terminal")
    _key(create=True)
    print(f"\nAPPROVE ONE ORDER\n  BUY {order['side'].upper()}  {order['ticker']}\n  quantity {int(order['quantity'])} contracts, limit {float(order['limit_price']) * 100:.0f}c, "
          f"maximum total cost ${float(order['max_cost']):.2f}\n  valid {ttl_min:g} minutes, one use")
    if input(f"Type APPROVE {order['ticker']} to confirm: ").strip() != f"APPROVE {order['ticker']}":
        raise PermissionError("not confirmed")
    return issue(con, order, ttl_min)


def verify(con, order: dict, token: str) -> tuple[bool, str]:
    r = con.execute("SELECT * FROM approvals WHERE token=?", (token,)).fetchone()
    if r is None:
        return False, "unknown approval"
    if r["used"]:
        return False, "approval already used"
    if datetime.fromisoformat(r["expires"]) < datetime.now(timezone.utc):
        return False, "approval expired"
    bound = dict(ticker=r["ticker"], side=r["side"], limit_price=r["limit_price"], quantity=r["quantity"], max_cost=r["max_cost"])
    if canonical(bound) != canonical(order):
        return False, "order differs from what was approved"
    try:
        good = hmac.compare_digest(r["sig"], _sign(order, token, r["expires"]))
    except PermissionError as e:
        return False, str(e)
    return (True, "ok") if good else (False, "signature does not match")


def place_approved(k, con, order: dict, token: str, cfg: dict, quote: dict, vetoes: list[str] | None = None) -> dict:
    """Send one approved order. `quote` = {'ask': price now for that side, 'age_s': seconds since it was read}."""
    if cfg.get("mode") != "live_review":
        raise PermissionError(f"mode is {cfg.get('mode')!r}: orders can only be sent in live_review mode")
    if not getattr(k, "authed", False):
        raise PermissionError("no Kalshi API key on this machine")
    ok, why = verify(con, order, token)
    if not ok:
        raise PermissionError(why)
    con.execute("UPDATE approvals SET used=? WHERE token=? AND used IS NULL", (store.now(), token))      # one use, before anything is sent
    con.commit()
    if vetoes:
        raise PermissionError("risk auditor: " + "; ".join(vetoes))
    if quote.get("age_s", 1e9) > cfg["quote_max_age_s"]:
        raise PermissionError("quote is stale")
    if quote["ask"] > float(order["limit_price"]) + 1e-9:
        raise PermissionError(f"price {quote['ask']:.2f} is above the approved limit {float(order['limit_price']):.2f}")
    from . import mathx as M
    q, lim = int(order["quantity"]), float(order["limit_price"])
    if q * lim + M.order_fee(lim, q) > float(order["max_cost"]) + 1e-9:
        raise PermissionError("order would cost more than the approved maximum")
    # Kalshi quotes everything from the YES side: buying NO at L is side 'ask' at 1 - L. Fill what is there, rest nothing.
    body = dict(ticker=order["ticker"], side="bid" if order["side"] == "yes" else "ask", count=f"{q:.2f}",
                price=f"{lim if order['side'] == 'yes' else 1 - lim:.4f}", time_in_force="immediate_or_cancel", client_order_id=token)
    return k.request("POST", "/portfolio/events/orders", json=body, auth=True)


def tickets(decisions: list[dict]) -> list[dict]:
    """Order tickets for accepted single positions, for the owner to review and approve one by one (LIVE REVIEW mode)."""
    out = []
    for d in decisions:
        if d["decision"] == "ACCEPT" and d["contracts"] > 0:
            lim = round(min(d["ask"] + 0.02, 0.99), 2)
            out.append(dict(ticker=d["ticker"], side=d["side"], limit_price=lim, quantity=int(d["contracts"]),
                            max_cost=round(d["contracts"] * lim * 1.03 + 0.01, 2), title=d["title"], reason=d["reason"]))
    return out
