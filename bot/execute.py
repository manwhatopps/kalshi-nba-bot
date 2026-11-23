"""Turn a suggested combo into a real Kalshi position via the RFQ flow:
create combo market -> RFQ -> read maker quotes -> accept only if still +EV -> verify fill.

Safety rails: dry-run unless --live; demo exchange unless KALSHI_ENV=prod; per-combo and
daily stake caps; minimum EV against the REAL quote; RFQs that we do not accept get deleted.
"""
from __future__ import annotations

import time

from . import config, ledger
from .kalshi import KalshiError, fee


def _quote_yes_price(q) -> float | None:
    yes = float(q.get("yes_bid_dollars") or 0)
    no = float(q.get("no_bid_dollars") or 0)
    if yes <= 0:
        return None  # maker declined the YES side
    # Worst case of the two possible readings of a quote: never pay more than we checked.
    return max(yes, 1 - no) if no > 0 else yes


def place(k, combos, collection: str, run_id: str, live: bool, min_ev: float = 0.05,
          max_rfqs: int = 3, wait_s: float = 12):
    if not k.authed:
        print("\nNo Kalshi key configured -> cannot request quotes. Set KALSHI_API_KEY_ID and "
              "KALSHI_PRIVATE_KEY_PATH in .env (see README).")
        return
    mode = f"{'LIVE' if live else 'DRY-RUN (quotes only)'} on {config.KALSHI_ENV.upper()}"
    print(f"\n=== RFQ {mode} | collection {collection} ===")
    budget = config.MAX_DAILY_STAKE - ledger.placed_today()
    for c in combos[:max_rfqs]:
        stake = c.stake()
        if stake < 0.5:
            print(f"  skip (Kelly stake ${stake} too small): {c.legs[0].title} ...")
            continue
        if live and stake > budget:
            print(f"  daily cap reached (${config.MAX_DAILY_STAKE}); stopping")
            break
        legs = [{"market_ticker": l.ticker, "event_ticker": l.event_ticker, "side": "yes"} for l in c.legs]
        try:
            mk = k.create_combo_market(collection, legs)
            ticker = mk.get("market_ticker") or mk["market"]["ticker"]
            rfq = k.create_rfq(ticker, stake)
        except KalshiError as e:
            print(f"  ! could not create combo/RFQ: {e}")
            continue
        best, best_q = None, None
        t0 = time.time()
        while time.time() - t0 < wait_s:
            for q in k.quotes(rfq):
                if q.get("status") not in (None, "open", "active"):
                    continue
                p = _quote_yes_price(q)
                if p and (best is None or p < best):
                    best, best_q = p, q
            time.sleep(1.0)
        if best is None:
            print(f"  no quotes in {wait_s:.0f}s for {ticker}")
            k.delete_rfq(rfq)
            ledger.log_combo(run_id, c, "no_quote", combo_ticker=ticker)
            continue
        ev = c.p_fair / (best + fee(best)) - 1
        print(f"  quote {best:.4f} ({1 / best:.1f}x) vs est {c.est_price:.4f} | fair {c.p_fair:.4f} | EV {ev:+.1%}")
        if not live or ev < min_ev:
            k.delete_rfq(rfq)
            ledger.log_combo(run_id, c, "quoted", quote_price=best, combo_ticker=ticker,
                             note="" if live else "dry-run")
            continue
        try:
            k.accept_quote(rfq, best_q["id"], "yes")
        except KalshiError as e:
            print(f"  ! accept failed: {e}")
            ledger.log_combo(run_id, c, "accept_failed", quote_price=best, combo_ticker=ticker)
            continue
        time.sleep(3)
        fills = k.fills(ticker)
        sides = {f.get("purchased_side") or f.get("side") for f in fills}
        note = "" if not fills else f"fill sides={sides}"
        if fills and "yes" not in sides:
            note += " !!! FILLED ON NO SIDE - check position and close it"
            print("  !!! fill was NOT on the YES side. Check your Kalshi portfolio now.")
        ledger.log_combo(run_id, c, "placed", quote_price=best, stake=stake, combo_ticker=ticker, note=note)
        budget -= stake
        print(f"  PLACED ${stake} on {ticker} {note}")
