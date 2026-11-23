"""Slips: tonight's plan turned into orders that can actually be placed on Kalshi.

A pick is not a bet until it names the exact market and side, the dollars and contracts, the worst
price worth paying, how long that is good for, and passes Kalshi's own rules for what may be combined.
This module adds those things to what the rest of the bot already decides:

  CORE slips    single bets from the fund's graded rule (today: NBA double-double UNDER, in-game, 35-80c).
                Each names the market, BUY NO, a limit price, the contract count for the fund's stake,
                when to place it, and the rule's own record in that price band.
  TICKET slips  the degen-mode lottery cards as Kalshi combos: legs in the exact form Kalshi's combo
                endpoint takes, checked against the combo collection's rules (which events can be
                combined, which are YES-only, how many legs per event), with a walk-away payout.

  python -m bot slip            build tonight's slips, print them, save data/slips/<date>.json + .html
  python -m bot slip --check    re-price a saved slate just before placing (started games, moved prices)
  python -m bot slip --quote    needs the owner's API key: create each combo on Kalshi and ask for a
                                real quote; prints the payout offered next to the walk-away line.
                                Adding --live accepts a quote only if it beats that line AND fits the
                                MAX_STAKE_PER_COMBO / MAX_DAILY_STAKE caps. Nothing is placed without it.
"""
from __future__ import annotations

import html
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config, fund, ledger
from .kalshi import KalshiError, data_client, fee

COLLECTION = "KXMVECROSSCATEGORY-R"
DIR = config.DATA_DIR / "slips"
ET = ZoneInfo("America/New_York")
SITE = "https://kalshi.com/markets"
DD_LO, DD_HI = 0.35, 0.80          # the price range the double-double rule was tested on
SLIPPAGE = 0.02                    # a single bet may pay this much more than the price seen, never past the range


# ---------------------------------------------------------------- Kalshi's combo rules
def collection_rules(k, collection: str = COLLECTION) -> dict:
    """{event_ticker: (yes_only, max_legs_from_this_event or None)} for everything Kalshi lets you combine."""
    for c in k.collections():
        if c.get("collection_ticker") == collection:
            return {e["ticker"]: (bool(e.get("is_yes_only")), e.get("size_max")) for e in c.get("associated_events") or []}
    return {}


def legal(legs: list[dict], rules: dict) -> list[str]:
    """Reasons Kalshi would refuse this combo (empty list = it can be built)."""
    if not rules:
        return ["could not load Kalshi's combo rules, so this card is unchecked"]
    out, per = [], Counter()
    if len(legs) < 2:
        out.append("a combo needs at least 2 legs")
    for l in legs:
        r = rules.get(l["event_ticker"])
        if r is None:
            out.append(f"{l['ticker']}: not offered in combos")
            continue
        if r[0] and l["side"] != "yes":
            out.append(f"{l['ticker']}: this market can only be combined as YES")
        per[l["event_ticker"]] += 1
        if r[1] and per[l["event_ticker"]] > r[1]:
            out.append(f"{l['event_ticker']}: at most {r[1]} leg(s) from this market group")
    return out


# ---------------------------------------------------------------- wording, links, sizes
def pick_text(title: str, side: str) -> str:
    """'Patrick Williams: 15+ points' bought NO -> 'Patrick Williams UNDER 15 points'."""
    m = re.match(r"^(.*?): (\d+)\+ (.*)$", title or "")
    if m:
        return f"{m[1]} UNDER {m[2]} {m[3]}" if side == "no" else f"{m[1]} {m[2]}+ {m[3]}"
    name, _, what = (title or "").partition(": ")
    return f"{name}: NO {what.lower()}" if side == "no" else f"{name}: {what.lower()}"


def tap_text(title: str, side: str) -> str:
    """What to press in the Kalshi app for this leg."""
    return f'{side.upper()} on "{title}"'


def market_url(ticker: str) -> str:
    """Event page with this market pre-selected. Format taken from open-source Kalshi clients (the middle
    word is a label the site replaces); kalshi.com blocks automated requests, so it is not tested from here."""
    series, event = ticker.split("-")[0], "-".join(ticker.split("-")[:2])
    return f"{SITE}/{series.lower()}/m/{event.lower()}?op_market_ticker={ticker}"


def contracts_for(stake: float, price: float) -> tuple[int, float, float]:
    """(contracts, total cost with fee, profit if it wins) for a stake at a limit price."""
    n = int(stake // fund.cost_with_fee(price)) if price > 0 else 0
    while n > 0 and n * price + math.ceil(7 * n * price * (1 - price) - 1e-9) / 100 > stake + 1e-9:
        n -= 1
    cost = round(n * price + math.ceil(7 * n * price * (1 - price) - 1e-9) / 100, 2) if n else 0.0
    return n, cost, round(n - cost, 2)


def _et(ts) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(ET).strftime("%-I:%M %p ET").lower().replace("et", "ET")


def matchups(k, events: list[str]) -> dict:
    """{game code: 'Cleveland vs Chicago'} from one event per game."""
    out = {}
    for ev in events:
        g = ev.split("-")[1]
        if g in out:
            continue
        try:
            out[g] = (k.request("GET", f"/events/{ev}")["event"].get("title") or "").split(":")[0].strip() or g
        except Exception:
            out[g] = g
    return out


# ---------------------------------------------------------------- core slips (single bets)
def band_record() -> dict:
    """The double-double rule's record since the walk-forward logic switched it on, by price band."""
    d = fund.dd_under_bets()
    if d.empty:
        return {}
    d = d[d.date >= fund.DD_ACTIVATED]
    out = {}
    for b, g in d.groupby("band"):
        c = g.cost + fund.TAKER_FEE * g.cost * (1 - g.cost)
        out[b] = dict(bets=len(g), won=float(g.win.mean()), needs=float(c.mean()), ret=float((g.win - c).sum() / c.sum()))
    return out


def _band(p: float) -> str:
    return "0.35-0.50" if p < 0.50 else "0.50-0.65" if p < 0.65 else "0.65-0.80"


def core_slips(k, cfg: dict, hours: float) -> dict:
    from . import watch
    from .history import START_OFFSET
    rule, stake = watch.fund_stake(k)
    core = cfg["core_pct"] * cfg["bankroll"]
    out = dict(rule=rule, stake=stake, sleeve=core, slips=[], waiting=0, taken=0, note="")
    if rule != "dd_under" or stake <= 0:
        out["note"] = ("No rule is graded A or B right now, so the core sleeve stays in cash tonight."
                       if not rule else f"Rule '{rule}' is graded but has no slip writer yet.")
        return out
    rec, now = band_record(), datetime.now(timezone.utc)
    taken = {t for (r, t, s) in watch._seen() if r == "dd_under"}
    rows = []
    for m in k.markets("KXNBA2D"):
        occ = m.get("occurrence_datetime")
        if not occ or "-" not in m.get("event_ticker", ""):
            continue
        tip = datetime.fromisoformat(occ.replace("Z", "+00:00")) - START_OFFSET["NBA"]
        mins = (now - tip).total_seconds() / 60
        if mins > 170 or tip > now + timedelta(hours=hours):
            continue
        bid = float(m.get("yes_bid_dollars") or 0)
        if bid <= 0:
            continue
        price, size = round(1 - bid, 2), float(m.get("yes_bid_size_fp") or 0)
        if m["ticker"] in taken:
            out["taken"] += 1
            continue
        if not (DD_LO <= price < DD_HI):
            out["waiting"] += 1
            continue
        rows.append(dict(ticker=m["ticker"], event_ticker=m["event_ticker"], title=m.get("title", ""), tipoff=tip,
                         live=mins >= 0, price=price, size=size))
    rows.sort(key=lambda r: (not r["live"], r["tipoff"], -r["size"]))
    names = matchups(k, [r["event_ticker"] for r in rows])
    room = int(core // stake) if stake else 0
    for i, r in enumerate(rows, 1):
        limit = round(min(r["price"] + SLIPPAGE, DD_HI - 0.01), 2)
        n, cost, profit = contracts_for(stake, limit)
        b = rec.get(_band(r["price"]), {})
        game = r["event_ticker"].split("-")[1]
        out["slips"].append(dict(
            id=f"C{i}", sleeve="core", kind="single", rule="dd_under", status="open" if i <= room else "reserve",
            ticker=r["ticker"], event_ticker=r["event_ticker"], side="no", action="BUY NO",
            pick=pick_text(r["title"], "no"), tap=tap_text(r["title"], "no"), matchup=names.get(game, game),
            tipoff=r["tipoff"].isoformat(), live=r["live"],
            when=("now: the game is on and the price is in range" if r["live"]
                  else f"from tip-off ({_et(r['tipoff'])}), if NO is still offered at {DD_LO * 100:.0f}-{DD_HI * 100:.0f}c"),
            price_seen=r["price"], limit=limit, size_shown=r["size"], stake=stake, contracts=n, cost=cost,
            pays=float(n), profit=profit, band=_band(r["price"]), record=b, url=market_url(r["ticker"]),
            # Kalshi's order endpoint (POST /portfolio/events/orders) quotes everything from the YES side:
            # buying NO at the limit is side "ask" at 1 - limit. Fill what is there now, leave nothing resting.
            order=dict(ticker=r["ticker"], side="ask", count=f"{n:.2f}", price=f"{1 - limit:.4f}",
                       time_in_force="immediate_or_cancel")))
    return out


# ---------------------------------------------------------------- ticket slips (combos)
def ticket_slips(k, cfg: dict, hours: float, rules: dict, tickets=None) -> list[dict]:
    from . import degen
    tickets = [tuple(t) for t in (tickets or cfg.get("degen_mode", [[10, 1000], [5, 1000], [5, 5000]]))]
    c = degen.candidates(k, hours=hours, rules=rules)
    if c.empty:
        return []
    names = matchups(k, list(c.event_ticker.unique()))
    used, out = set(), []
    for n, (stake, goal) in enumerate(tickets, 1):
        need = goal / stake
        card = degen.build(c, need, skip=used, seed=n)
        if card is None:
            out.append(dict(id=f"T{n}", sleeve="degen", kind="combo", status="unbuilt", stake=stake, target=goal, need_x=need,
                            legs=[], problems=["not enough different legs left tonight"]))
            continue
        used |= set(card.ticker)
        card = card.sort_values(["tipoff", "game"])
        legs = [dict(ticker=r.ticker, event_ticker=r.event_ticker, side=r.side, title=r.title, pick=pick_text(r.title, r.side),
                     tap=tap_text(r.title, r.side), matchup=names.get(r.game, r.game), game=r.game, tipoff=r.tipoff.isoformat(),
                     price=round(float(r.cost), 2), mid=round(float(r.mid), 4), p_hat=round(float(r.p_hat), 4),
                     chain_for=int(r.pro), chain_against=int(r.con), why=[w for w in str(r.why).split(" | ") if w],
                     url=market_url(r.ticker)) for r in card.itertuples()]
        mid, p = float(np.prod(card.mid)), float(np.prod(card.p_hat))
        out.append(dict(
            id=f"T{n}", sleeve="degen", kind="combo", status="open", stake=stake, target=goal, need_x=need,
            max_price=round(1 / need, 6), fair_x=1 / mid, p_hit=p, one_in=1 / p, exp_loss=round(stake * (1 - need * p), 2),
            n_legs=len(legs), n_games=int(card.game.nunique()), expires=min(l["tipoff"] for l in legs),
            collection=COLLECTION, legs=legs, problems=legal(legs, rules),
            payload=dict(selected_markets=[dict(market_ticker=l["ticker"], event_ticker=l["event_ticker"], side=l["side"]) for l in legs])))
    return out


# ---------------------------------------------------------------- build, save, load
def build(k, hours: float = 14, tickets=None) -> dict:
    cfg = fund.load_cfg()
    rules = {}
    try:
        rules = collection_rules(k)
    except Exception as e:
        print(f"  ! combo rules unavailable: {str(e)[:100]}")
    now = datetime.now(timezone.utc)
    doc = dict(made_at=now.isoformat(timespec="seconds"), slate=(now.astimezone(ET) - timedelta(hours=4)).date().isoformat(),
               bankroll=cfg["bankroll"], hours=hours, collection=COLLECTION, rules_loaded=len(rules))
    doc["core"] = core_slips(k, cfg, hours)
    doc["tickets"] = ticket_slips(k, cfg, hours, rules, tickets)
    return doc


def save(doc: dict) -> tuple:
    DIR.mkdir(parents=True, exist_ok=True)
    j, h = DIR / f"{doc['slate']}.json", DIR / f"{doc['slate']}.html"
    j.write_text(json.dumps(doc, indent=1, default=str))
    page = to_html(doc)
    h.write_text(page)
    (DIR / "latest.html").write_text(page)      # stable path, so the published page keeps one address
    return j, h


def load(slate: str | None = None) -> dict | None:
    fs = sorted(DIR.glob("2*.json")) if DIR.exists() else []
    f = DIR / f"{slate}.json" if slate else (fs[-1] if fs else None)
    return json.loads(f.read_text()) if f and f.exists() else None


# ---------------------------------------------------------------- text
def _money(v: float) -> str:
    return f"${v:,.2f}" if abs(v) < 100 and v != int(v) else f"${v:,.0f}"


def to_text(doc: dict, why: bool = False) -> str:
    L, core, made = [], doc["core"], pd.Timestamp(doc["made_at"])
    spend = sum(t["stake"] for t in doc["tickets"] if t["status"] == "open")
    L.append(f"\nSLIPS for {pd.Timestamp(doc['slate']):%a %b %-d}   made {_et(made)}   bankroll {_money(doc['bankroll'])}")
    L.append("Paper slips: nothing here has been placed. Prices move; run `python -m bot slip --check` right before placing.")
    L.append(f"\nCORE: single bets, {_money(core['stake'])} each, at most {_money(core['sleeve'])} at risk at once")
    if core["note"]:
        L.append(f"  {core['note']}")
    elif not core["slips"]:
        L.append(f"  No double-double UNDER is in the 35-80c range right now ({core['waiting']} markets outside it, "
                 f"{core['taken']} already taken by the live watcher). The watcher issues these as games go on.")
    for s in core["slips"]:
        r = s["record"]
        L.append(f"\n  {s['id']}  {s['action']}  {s['pick']}   [{s['matchup']}]" + ("   (over the sleeve cap: only if an earlier slip cannot be filled)" if s["status"] == "reserve" else ""))
        L.append(f"      when   {s['when']}")
        L.append(f"      order  {s['contracts']} contracts, limit {s['limit'] * 100:.0f}c (seen {s['price_seen'] * 100:.0f}c, {s['size_shown']:.0f} offered)  "
                 f"= {_money(s['cost'])} with fee; pays {_money(s['pays'])} if it wins (+{_money(s['profit'])})")
        if r:
            L.append(f"      why    this price band since {fund.DD_ACTIVATED}: {r['bets']} bets, won {r['won']:.1%}, needed {r['needs']:.1%} -> {r['ret']:+.1%} per $1")
        L.append(f"      market {s['ticker']}   {s['url']}")
    if core["slips"]:
        L.append(f"\n  ({core['waiting']} more double-double markets are outside the range now and may enter it during play; {core['taken']} already taken.)")
    L.append(f"\nTICKETS: lottery cards, {_money(spend)} tonight. Every one has a negative expected value; buy only at the stated payout or better.")
    if not doc["tickets"]:
        L.append("  Nothing qualifies right now (props usually post the morning of the game).")
    for t in doc["tickets"]:
        if t["status"] == "unbuilt":
            L.append(f"\n  {t['id']}  {_money(t['stake'])} to win {_money(t['target'])}: {t['problems'][0]}")
            continue
        L.append(f"\n  {t['id']}  {_money(t['stake'])} to win {_money(t['target'])}   {t['n_legs']} legs, {t['n_games']} games   good until {_et(t['expires'])}")
        L.append(f"      PLACE ONLY IF Kalshi shows a payout of {_money(t['target'])} or more for {_money(t['stake'])} "
                 f"({t['need_x']:,.0f}x; combo price {t['max_price'] * 100:.2f}c or less).")
        L.append(f"      Legs alone are worth {t['fair_x']:,.0f}x.  Chance it hits: about 1 in {t['one_in']:,.0f}.  "
                 f"Expected loss at the walk-away payout: {_money(t['exp_loss'])}.")
        L.append("      Kalshi's combo rules: " + ("OK, this card can be built as one combo" if not t["problems"] else "PROBLEM: " + "; ".join(t["problems"][:3])))
        g = None
        for l in t["legs"]:
            if l["matchup"] != g:
                g = l["matchup"]
                L.append(f"      {g}  (tips {_et(l['tipoff'])})")
            L.append(f"        [ ] {l['pick']:44s} {l['tap']:46s} {l['price'] * 100:.0f}c")
            if why:
                L.extend(f"              - {w}" for w in l["why"])
    return "\n".join(L)


# ---------------------------------------------------------------- re-check before placing
def quotes_now(k, tickers: list[str]) -> dict:
    out = {}
    for i in range(0, len(tickers), 100):
        d = k.request("GET", "/markets", params={"tickers": ",".join(tickers[i:i + 100]), "limit": 100})
        for m in d.get("markets") or []:
            out[m["ticker"]] = dict(bid=float(m.get("yes_bid_dollars") or 0), ask=float(m.get("yes_ask_dollars") or 0),
                                    status=m.get("status"), result=m.get("result") or "")
    return out


def check(k, doc: dict) -> str:
    """Is each saved slip still good to place right now?"""
    now = datetime.now(timezone.utc)
    tick = [s["ticker"] for s in doc["core"]["slips"]] + [l["ticker"] for t in doc["tickets"] for l in t.get("legs", [])]
    q = quotes_now(k, sorted(set(tick))) if tick else {}
    L = [f"\nSLIP CHECK  {_et(now)}   (slips made {_et(doc['made_at'])})"]
    for s in doc["core"]["slips"]:
        m = q.get(s["ticker"])
        price = round(1 - m["bid"], 2) if m and m["bid"] > 0 else None
        if not m or m["status"] != "active" or m["result"]:
            v = "CLOSED: drop it"
        elif price is None:
            v = "NO OFFER right now: wait"
        elif price > s["limit"]:
            v = f"PRICE ROSE to {price * 100:.0f}c (limit {s['limit'] * 100:.0f}c): " + ("still inside the tested range, re-run to resize" if price < DD_HI else "outside the tested range, skip")
        elif price < DD_LO:
            v = f"PRICE FELL to {price * 100:.0f}c, below the tested range: skip"
        else:
            v = f"GOOD at {price * 100:.0f}c" + ("" if now >= datetime.fromisoformat(s["tipoff"]) else f" (place from tip-off, {_et(s['tipoff'])})")
        s["check"] = v
        L.append(f"  {s['id']}  {s['pick'][:44]:44s} {v}")
    for t in doc["tickets"]:
        if not t.get("legs"):
            continue
        started, gone, mids = [], [], []
        for l in t["legs"]:
            m = q.get(l["ticker"])
            if not m or m["status"] != "active" or m["result"]:
                gone.append(l["pick"])
                continue
            if now >= datetime.fromisoformat(l["tipoff"]):
                started.append(l["pick"])
            mid = (m["bid"] + m["ask"]) / 2 if m["ask"] > 0 else m["bid"]
            mids.append(1 - mid if l["side"] == "no" else mid)
        fair = 1 / float(np.prod(mids)) if mids and not gone else float("nan")
        if gone or started:
            v = f"DEAD: {len(started)} leg(s) already playing, {len(gone)} closed. Build a new card (`python -m bot slip`)."
        elif fair < t["need_x"]:
            v = f"SKIP: the legs are now worth only {fair:,.0f}x, less than the {t['need_x']:,.0f}x this ticket needs."
        else:
            v = (f"GOOD: legs now worth {fair:,.0f}x (were {t['fair_x']:,.0f}x). Still place only at {_money(t['target'])} or more "
                 f"for {_money(t['stake'])}; a quote at the walk-away line is {t['need_x'] / fair:.0%} of fair value.")
        t["check"] = v
        L.append(f"  {t['id']}  {_money(t['stake'])} -> {_money(t['target'])}   {v}")
    doc["checked_at"] = now.isoformat(timespec="seconds")
    return "\n".join(L)


# ---------------------------------------------------------------- real quotes (owner's key required)
def quote(k, doc: dict, live: bool = False, wait_s: float = 12) -> None:
    """Create each ticket as a Kalshi combo market and ask market makers for a price. With `live`, accept a
    quote only when it pays the target or more and the stake fits the configured caps."""
    from .execute import _quote_yes_price
    if not k.authed:
        print("\nNo Kalshi key on this machine, so no combo can be created or quoted. Put KALSHI_API_KEY_ID and "
              "KALSHI_PRIVATE_KEY_PATH in .env on a machine you control (see README), then re-run with --quote.")
        return
    print(f"\nQUOTES ({'LIVE' if live else 'look only'}, {config.KALSHI_ENV.upper()})")
    budget = config.MAX_DAILY_STAKE - ledger.placed_today()
    for t in doc["tickets"]:
        if t["status"] != "open" or t["problems"]:
            continue
        try:
            mk = k.create_combo_market(t["collection"], t["payload"]["selected_markets"])
            ticker = mk.get("market_ticker") or mk["market"]["ticker"]
            rfq = k.create_rfq(ticker, t["stake"])
        except KalshiError as e:
            print(f"  {t['id']}: could not create the combo or request a quote: {str(e)[:160]}")
            continue
        t["combo_ticker"], t["combo_url"] = ticker, market_url(ticker)
        best, best_q, t0 = None, None, time.time()
        while time.time() - t0 < wait_s:
            for qd in k.quotes(rfq):
                p = _quote_yes_price(qd)
                if p and (best is None or p < best):
                    best, best_q = p, qd
            time.sleep(1.0)
        if best is None:
            print(f"  {t['id']}: no market maker quoted in {wait_s:.0f}s. Combo: {t['combo_url']}")
            k.delete_rfq(rfq)
            continue
        pays = t["stake"] / (best + fee(best))
        good = pays >= t["target"]
        t["quote_price"], t["quote_pays"] = best, round(pays, 2)
        print(f"  {t['id']}: quote {best * 100:.2f}c -> {_money(t['stake'])} pays {_money(pays)} (walk-away {_money(t['target'])}): "
              f"{'MEETS the line' if good else 'BELOW the line, skip'}   {t['combo_url']}")
        ok_caps = t["stake"] <= config.MAX_STAKE_PER_COMBO and t["stake"] <= budget
        if not (live and good and ok_caps):
            if live and good and not ok_caps:
                print(f"      not placed: stake {_money(t['stake'])} is over the caps (per combo {_money(config.MAX_STAKE_PER_COMBO)}, "
                      f"left today {_money(max(budget, 0))}). The owner sets those in .env.")
            k.delete_rfq(rfq)
            continue
        try:
            k.accept_quote(rfq, best_q["id"], "yes")
            budget -= t["stake"]
            t["status"] = "placed"
            print(f"      PLACED {_money(t['stake'])} on {ticker}")
        except KalshiError as e:
            print(f"      accept failed: {str(e)[:160]}")


# ---------------------------------------------------------------- phone page
def to_html(doc: dict) -> str:
    from .slip_page import render
    return render(doc)


# ---------------------------------------------------------------- command
def run(args) -> None:
    k = data_client()
    if args.check or args.quote:
        doc = load(args.slate)
        if doc is None:
            print("No saved slips yet: run `python -m bot slip` first.")
            return
        if args.check:
            print(check(k, doc))
        if args.quote:
            from .kalshi import trade_client
            quote(trade_client(), doc, live=args.live)
        save(doc)
        return
    doc = build(k, hours=args.hours, tickets=json.loads(args.tickets) if args.tickets else None)
    print(to_text(doc, why=args.why))
    j, h = save(doc)
    print(f"\nSaved: {j}  and  {h}")
    if args.log:
        for t in doc["tickets"]:
            if t["status"] != "open":
                continue
            card = pd.DataFrame(t["legs"])
            ledger.log_card(doc["made_at"][:10].replace("-", "") + t["id"], card, f"degen_{int(t['target'])}", t["p_hit"],
                            1 / t["need_x"], stake=t["stake"], note=f"slip {t['stake']}->{t['target']}")
