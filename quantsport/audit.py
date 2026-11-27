"""Settlement and the performance audit: what happened, whether our probabilities were any good, and what it cost.

Nothing here retrains anything. It grades: paper positions (P&L by tier), chain decisions (our probability against
the outcome and against the market's own), and every scanned side (calibration by sport and market type).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bot import config as botcfg

from . import store

RESULTS = botcfg.DATA_DIR / "qs_results.parquet"


def results_for(k, tickers: list[str]) -> dict:
    """{ticker: 'yes' | 'no'} for settled markets, cached on disk so each market is looked up once."""
    cache = pd.read_parquet(RESULTS).set_index("ticker").result.to_dict() if RESULTS.exists() else {}
    need = sorted(set(tickers) - set(cache))
    for i in range(0, len(need), 100):
        try:
            d = k.request("GET", "/markets", params={"tickers": ",".join(need[i:i + 100]), "limit": 100})
        except Exception as e:
            print(f"  ! results lookup failed: {str(e)[:80]}")
            break
        for m in d.get("markets") or []:
            if m.get("result") in ("yes", "no"):
                cache[m["ticker"]] = m["result"]
    if cache:
        pd.DataFrame({"ticker": list(cache), "result": list(cache.values())}).to_parquet(RESULTS, index=False)
    return cache


def grade_position(kind: str, legs: list[str], res: dict) -> tuple[str, int]:
    """('won' | 'lost' | 'open', legs hit). A card loses as soon as one leg loses and wins only when every leg has won."""
    hit = known = 0
    lost = False
    for leg in legs:
        t, side = leg.rsplit(":", 1)
        r = res.get(t)
        if r is None:
            continue
        known += 1
        if r == side:
            hit += 1
        else:
            lost = True
    return ("lost" if lost else "won" if known == len(legs) else "open"), hit


def settle(k, con) -> dict:
    pos = store.frame(con, "SELECT * FROM positions WHERE status='open'")
    dec = store.frame(con, "SELECT id, ticker, side FROM decisions WHERE result=''")
    tick = {l.rsplit(":", 1)[0] for row in pos.tickers for l in json.loads(row)} | set(dec.ticker)
    res = results_for(k, sorted(tick)) if tick else {}
    n_pos = n_dec = 0
    for r in pos.itertuples():
        status, hit = grade_position(r.kind, json.loads(r.tickers), res)
        if status == "open":
            continue
        pnl = (r.pays - r.stake) if status == "won" else -r.stake
        con.execute("UPDATE positions SET status=?, closed=?, pnl=? WHERE id=?", (status, store.now(), float(pnl), r.id))
        if r.kind == "combo":
            con.execute("UPDATE cards SET result=?, pnl=?, legs_hit=? WHERE card_id=?", (status, float(pnl), hit, r.ref))
        n_pos += 1
    for r in dec.itertuples():
        if r.ticker in res:
            con.execute("UPDATE decisions SET result=? WHERE id=?", ("won" if res[r.ticker] == r.side else "lost", r.id))
            n_dec += 1
    con.commit()
    return dict(positions=n_pos, decisions=n_dec)


def closing_prices(k, con) -> int:
    """For graded decisions without one: the last quote before the game started, for the side we looked at.
    Closing-line value = that closing price minus the ask at decision time (positive = the market moved our way)."""
    from datetime import datetime, timedelta, timezone
    from bot.history import _candles
    d = store.frame(con, "SELECT id, ticker, side FROM decisions WHERE result!='' AND close_price IS NULL AND decision IN ('WAIT','ACCEPT','VETO')")
    if d.empty:
        return 0
    start = {}
    tick = sorted(set(d.ticker))
    for i in range(0, len(tick), 100):
        try:
            for m in k.request("GET", "/markets", params={"tickers": ",".join(tick[i:i + 100]), "limit": 100}).get("markets") or []:
                occ = m.get("occurrence_datetime") or m.get("expected_expiration_time")
                if occ:
                    start[m["ticker"]] = datetime.fromisoformat(occ.replace("Z", "+00:00")) - timedelta(hours=3)
        except Exception as e:
            print(f"  ! closing prices: {str(e)[:80]}")
            return 0
    close, done = {}, 0
    by_hour = {}
    for t, st in start.items():
        by_hour.setdefault(st.replace(minute=0, second=0, microsecond=0), []).append(t)
    for hour, ts in sorted(by_hour.items()):
        for i in range(0, len(ts), 100):
            try:
                cs = _candles(k, ts[i:i + 100], hour - timedelta(hours=5), hour + timedelta(hours=1, minutes=5))
            except Exception as e:
                print(f"  ! candles: {str(e)[:80]}")
                continue
            for t, allc in cs.items():
                pre = [c for c in allc if c["end_period_ts"] <= start[t].timestamp() + 60]
                if pre:
                    b, a = pre[-1].get("yes_bid", {}).get("close_dollars"), pre[-1].get("yes_ask", {}).get("close_dollars")
                    if b is not None and a is not None:
                        close[t] = (float(b) + float(a)) / 2
    for r in d.itertuples():
        if r.ticker in close:
            con.execute("UPDATE decisions SET close_price=? WHERE id=?", (close[r.ticker] if r.side == "yes" else 1 - close[r.ticker], r.id))
            done += 1
    con.commit()
    return done


def shadow_record(con) -> dict:
    """The watch list's live record: every WAIT (and ACCEPT) the chain produced, graded as if bought at the ask it saw.
    One line per player, prop type and day, because a player's ladder is one position."""
    d = store.frame(con, "SELECT ts, ticker, side, title, decision, ask, fee, p_est, source, result, close_price FROM decisions "
                         "WHERE decision IN ('WAIT','ACCEPT') AND result IN ('won','lost')")
    if d.empty:
        return dict(n=0)
    parts = d.ticker.str.split("-")
    d["key"] = d.ts.str[:10] + "|" + parts.str[0] + "|" + parts.str[2].fillna("") + "|" + d.side
    d["mid_dist"] = (d.ask - 0.5).abs()
    one = d.sort_values("mid_dist").drop_duplicates("key")
    cost = one.ask + one.fee
    win = (one.result == "won").astype(float)
    net = win - cost
    # error bar by game, not by bet: several props in one game rise and fall together
    game = one.ticker.str.split("-").str[1]
    ret_ = float(net.sum() / cost.sum())
    resid = (net - ret_ * cost).groupby(game).sum()
    G = len(resid)
    se = float(np.sqrt(G / (G - 1) * (resid ** 2).sum()) / cost.sum()) if G > 1 else float("nan")
    clv = (one.close_price - one.ask).dropna()
    by = {src: dict(n=int(len(g)), won=float((g.result == "won").mean()), needed=float((g.ask + g.fee).mean()),
                    ret=float((((g.result == "won").astype(float)) - (g.ask + g.fee)).sum() / (g.ask + g.fee).sum())) for src, g in one.groupby("source")}
    return dict(n=int(len(one)), games=int(G), lines=int(len(d)), won=float(win.mean()), needed=float(cost.mean()), est=float(one.p_est.mean()),
                ret=float(net.sum() / cost.sum()), se=se, clv=float(clv.mean()) if len(clv) else None, clv_n=int(len(clv)), by_source=by)


def _score(p, y) -> dict:
    p, y = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4), np.asarray(y, float)
    return dict(brier=float(((p - y) ** 2).mean()), logloss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))


def calibration(k) -> pd.DataFrame:
    """Every scanned YES side that has settled: our probability and the market's against what happened."""
    fs = sorted(store.SCANS.glob("*.parquet")) if store.SCANS.exists() else []
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.side == "yes") & (d.start < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4))]
    d = d.sort_values("run_id").drop_duplicates("ticker", keep="last")              # the last look before the game
    if d.empty:
        return d
    res = results_for(k, list(d.ticker))
    d["result"] = d.ticker.map(res)
    d = d[d.result.isin(["yes", "no"])]
    d["y"] = (d.result == "yes").astype(float)
    return d


def report(k, con, verbose: bool = True) -> dict:
    out = {}
    pos = store.frame(con, "SELECT * FROM positions")
    done = pos[pos.status.isin(["won", "lost"])] if len(pos) else pos
    rows = []
    for tier, g in (done.groupby("tier") if len(done) else []):
        staked, pnl = float(g.stake.sum()), float(g.pnl.sum())
        rows.append(dict(tier=tier, positions=len(g), won=int((g.status == "won").sum()), staked=staked, pnl=pnl, roi=pnl / staked if staked else float("nan"),
                         verdict="too few to judge" if len(g) < 200 else "enough to read"))
    out["tiers"] = rows
    eq = store.frame(con, "SELECT * FROM equity ORDER BY ts")
    out["max_drawdown"] = float(eq.drawdown.max()) if len(eq) else 0.0
    cal = calibration(k)
    crow = []
    if len(cal):
        for (sport, kind), g in cal.groupby(["sport", "kind"]):
            if len(g) < 30:
                continue
            a, b = _score(g.p_est, g.y), _score(g.mid, g.y)
            crow.append(dict(sport=sport, market=kind, settled=len(g), brier_ours=a["brier"], brier_market=b["brier"], logloss_ours=a["logloss"],
                             logloss_market=b["logloss"], avg_p=float(g.p_est.mean()), hit=float(g.y.mean())))
    out["calibration"] = crow
    try:
        closing_prices(k, con)
    except Exception as e:
        print(f"  ! closing prices unavailable: {str(e)[:80]}")
    out["shadow"] = shadow_record(con)
    out["open"] = int((pos.status == "open").sum()) if len(pos) else 0
    if verbose:
        print("\nQUANTSPORT PERFORMANCE AUDIT (paper)")
        if not rows:
            print("  No paper position has settled yet.")
        for r in rows:
            print(f"  {r['tier']:12s} {r['positions']:4d} settled, {r['won']} won, staked ${r['staked']:.2f}, P&L {r['pnl']:+.2f} ({r['roi']:+.1%})  [{r['verdict']}]")
        print(f"  open positions: {out['open']}   maximum drawdown so far: {out['max_drawdown']:.1%}")
        sh = out["shadow"]
        if sh["n"]:
            print(f"\n  WATCH-LIST RECORD (as if every wait-listed bet had been bought at the ask it saw; one line per player, prop and day):")
            print(f"    {sh['n']} positions ({sh['lines']} lines), won {sh['won']:.1%}, needed {sh['needed']:.1%}, our estimate said {sh['est']:.1%} "
                  f"-> {sh['ret']:+.1%} per $1 (+/- {sh['se']:.1%}, {sh['games']} games)  [{'too few to judge' if sh['n'] < 200 or sh['games'] < 40 else 'enough to read'}]")
            for src, v in sh["by_source"].items():
                print(f"      {src}: {v['n']} positions, won {v['won']:.1%}, needed {v['needed']:.1%}, {v['ret']:+.1%}")
            if sh["clv"] is not None:
                print(f"    closing-line value on {sh['clv_n']} of them: the price for our side closed {sh['clv'] * 100:+.1f}c from the ask we saw "
                      "(positive = the market moved our way before the game; the close is a mid-price, so about -1c is no movement at all).")
        else:
            print("\n  Watch-list record: nothing wait-listed has settled yet.")
        if crow:
            print("\n  Our probability against the outcome, and against the market's own number (lower is better):")
            print(pd.DataFrame(crow).round(4).to_string(index=False))
            print("  Where our estimate IS the market price these match by construction; a gap appears only where a model or category moved the estimate.")
        else:
            print("  Calibration: no scanned market has settled yet.")
    return out
