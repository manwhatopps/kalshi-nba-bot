"""The dashboard: one self-contained page written from the latest run and the fund's database."""
from __future__ import annotations

import json
from html import escape as e

import pandas as pd

from bot.slip_page import CSS

from . import pipeline, store

EXTRA = """
table{border-collapse:collapse;width:100%;font-size:.86rem} th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} th{color:var(--muted);font-weight:600;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tiles div{background:var(--card);padding:10px 12px} .tiles b{display:block;font-family:var(--mono);font-size:1.05rem} .tiles span{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px} details+details{margin-top:8px}
summary{cursor:pointer;font-weight:600} details ol,details ul{margin:8px 0 4px;padding-left:20px;font-size:.86rem;color:var(--muted)} li{margin:2px 0;overflow-wrap:anywhere}
.nobet{border-left:4px solid var(--accent);background:var(--soft);padding:10px 12px;font-weight:600} .wrap{max-width:860px}
.stack{display:flex;flex-direction:column;gap:8px}
"""


def _m(v) -> str:
    return f"${v:,.2f}"


def _table(rows: list[dict], cols: list[tuple]) -> str:
    if not rows:
        return '<p class="muted small">Nothing yet.</p>'
    head = "".join(f"<th>{e(h)}</th>" for _, h, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{e(f(r.get(k)) if r.get(k) is not None else '')}</td>" for k, _, f in cols) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(o: dict, perf: dict, models: dict, positions: list[dict]) -> str:
    st, cfg = o["state"], o["cfg"]
    pct, num, txt = (lambda v: f"{v:+.1%}"), (lambda v: f"{v:,.0f}"), (lambda v: str(v))
    acc = [d for d in o["decisions"] if d["decision"] == "ACCEPT"]
    core = ('<p class="nobet">NO BET on the whole slate. Nothing has a positive expected value at the cautious end of our estimate after the fee.</p>'
            if not acc else "") + "".join(_decision(d) for d in sorted(o["decisions"], key=lambda d: (d["decision"] != "ACCEPT", -d["ret"]))[:8])
    cards = ""
    names = {"small": "Tier 2 · Small multipliers", "diversified": "Tier 3 · Diversified combos", "degen": "Tier 4 · Degen"}
    for tier in ("small", "diversified", "degen"):
        cs = [c for c in o["cards"] if c["tier"] == tier]
        cards += f'<h3 style="font-size:1rem;margin-top:8px">{names[tier]}</h3><p class="muted small">{_m(cfg["card_stake"][tier])} a card, pot {cfg["budget"][tier]:.0%} of bankroll, ' \
                 f'at most {cfg["weekly_spend"][tier]:.0%} a week. Buy only if the quote pays at least {cfg["min_keep"][tier]:.0%} of fair.' \
                 + (" Entertainment money: negative expected value by design." if tier == "degen" else "") + "</p>"
        cards += (f'<p class="muted small">{e(o["tier_notes"].get(tier, "No card."))}</p>' if not cs else '<div class="stack">' + "".join(_card(c) for c in cs) + "</div>")
    stack_rows = [dict(market=k, markets=v["markets"], model=v["model_logloss"], price=v["price_logloss"], weight=v["params"][2], z=v["z_model"],
                       verdict="beats the price" if v["helps"] else "does not beat the price") for k, v in sorted(models.get("stack", {}).items())]
    sim = o.get("sim") or {}
    return f"""<title>QuantSport Dashboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">
<style>{CSS}{EXTRA}</style>
<main class="wrap">
  <header><p class="tag">QUANTSPORT · {e(o['mode'].upper())} MODE · RUN {e(o['ts'][:16].replace('T', ' '))} UTC · MODEL {e(o['model_version'])}</p><h1>Fund dashboard</h1></header>
  <p class="notice">Paper only. No order has been placed and none can be without a signed, one-use approval from the owner at a terminal.</p>
  <h2>Fund</h2>
  <div class="tiles">
    <div><span>Bankroll</span><b>{_m(st['bankroll'])}</b></div><div><span>Available</span><b>{_m(st['cash'])}</b></div>
    <div><span>At risk</span><b>{_m(st['exposure'])}</b></div><div><span>Realized P/L</span><b>{st['realized']:+.2f}</b></div>
    <div><span>Drawdown</span><b>{st['drawdown']:.1%}</b></div><div><span>Risk scale</span><b>{o['state_before']['scale']:g}</b></div>
    <div><span>Open positions</span><b>{st['open']}</b></div><div><span>95% worst night</span><b>{_m(sim.get('var95', 0))}</b></div>
  </div>
  <p class="muted small">{e(o['guard'])}{' · STOPPED: ' + e('; '.join(o['state_before']['stops'])) if o['state_before']['stops'] else ''}</p>
  <h2>Market scanner</h2>
  <p class="muted small">{o['scanned']:,} buyable sides in {o['games']} games; {o['identified']} games matched to the rating models. "Best return" is the best expected return per $1 at our estimate after the fee.</p>
  {_table(o['summary'], [("sport", "Sport", txt), ("kind", "Market", txt), ("sides", "Sides", num), ("games", "Games", num), ("best_ret", "Best return", pct)])}
  <h2>Tier 1 · Core singles</h2>
  <p class="muted small">A single is accepted only if its expected value is positive at {cfg['z_edge']:g} standard errors below our estimate, after the fee and the real order book, and the auditor finds no limit breached. Sized at {cfg['kelly_fraction']:g} × Kelly, capped at {cfg['kelly_cap']:.0%}.</p>
  <div class="stack">{core}</div>
  <h2>Cards</h2>
  <p class="muted small">One leg per game on a card and no game on two cards. Chances multiply the legs' probabilities; legs in different games are independent in the data. Walk-away is the payout a real quote must reach.</p>
  {cards}
  <h2>Portfolio</h2>
  <p class="muted small">If every pending card were bought at its walk-away payout: expected {sim.get('expected', 0):+.2f}, losing night {sim.get('p_loss', 0):.0%} of the time, average of the worst 5% {_m(sim.get('cvar95', 0))}.
  Limits: {cfg['limit']['event']:.0%} of bankroll per game, {cfg['limit']['team']:.0%} per team, {cfg['limit']['player']:.0%} per player, {cfg['limit']['sport']:.0%} per sport; stops at {cfg['max_daily_loss']:.0%} daily, {cfg['max_weekly_loss']:.0%} weekly, {cfg['max_drawdown']:.0%} drawdown.</p>
  {_positions(positions)}
  <h2>Performance</h2>
  {_table(perf.get('tiers', []), [("tier", "Tier", txt), ("positions", "Settled", num), ("won", "Won", num), ("staked", "Staked", _m), ("pnl", "P/L", lambda v: f"{v:+.2f}"), ("roi", "ROI", pct), ("verdict", "Sample", txt)])}
  {_shadow(perf.get('shadow') or {})}
  <p class="muted small" style="margin-top:8px">Our probability against the outcome, and against the market's own (Brier score, lower is better). Maximum drawdown so far {perf.get('max_drawdown', 0):.1%}.</p>
  {_table(perf.get('calibration', []), [("sport", "Sport", txt), ("market", "Market", txt), ("settled", "Settled", num), ("brier_ours", "Ours", lambda v: f"{v:.4f}"), ("brier_market", "Market", lambda v: f"{v:.4f}"), ("hit", "Hit rate", lambda v: f"{v:.1%}"), ("avg_p", "Avg estimate", lambda v: f"{v:.1%}")])}
  <h2>Do the models beat the price?</h2>
  <p class="muted small">Rating model against Kalshi's pre-game price on settled markets (log-loss, lower is better). "Weight" is what a fitted blend puts on the model after the price. The fund uses a model only where it beats the price on later data.</p>
  {_table(stack_rows, [("market", "Sport : market", txt), ("markets", "Markets", num), ("model", "Model", lambda v: f"{v:.3f}"), ("price", "Price", lambda v: f"{v:.3f}"), ("weight", "Weight", lambda v: f"{v:+.2f}"), ("z", "z", lambda v: f"{v:+.1f}"), ("verdict", "Verdict", txt)])}
  <h2>Where the model disagrees with the price</h2>
  <p class="muted small">Context only. The model has not beaten the price, so these are not bets.</p>
  {_table(o.get('disagreements', []), [("title", "Market", txt), ("sport", "Sport", txt), ("market", "Price", lambda v: f"{v:.0%}"), ("model", "Model", lambda v: f"{v:.0%}"), ("gap", "Gap", pct)])}
</main>
"""


def _positions(rows: list[dict]) -> str:
    """Open cards with where they stand now; the hedge appears when one leg is left."""
    if not rows:
        return '<p class="muted small">No open positions.</p>'
    if "legs_left" not in rows[0]:                                   # live status unavailable: the plain list
        return _table(rows, [("label", "Open position", str), ("tier", "Tier", str), ("stake", "Stake", _m), ("pays", "Pays", _m)])
    out = []
    for r in rows:
        state = "dead: a leg lost" if r["dead"] else "no live price" if r["value"] is None else f'{r["alive"]:.0%} to cash · worth about {_m(r["value"])}'
        legs = "".join(f'<li>{"✗ lost" if l["state"] == "lost" else "to play"} · {e(l["side"].upper())} {e((l.get("title") or l["ticker"])[:70])}'
                       + (f' · now {l["mid"]:.0%}' if l["state"] == "pending" and l.get("mid") is not None else "") + "</li>"
                       for l in r["legs"] if l["state"] == "lost" or (l["state"] == "pending" and not r["dead"]))
        hedge = ""
        if r.get("hedge"):
            h = r["hedge"]
            hedge = (f'<p class="small" style="margin-top:6px"><b>Hedge</b> with {e(h["opposite"].upper())} on the last leg at {h["opposite_cost"] * 100:.0f}¢ a contract (fee included): '
                     + "; ".join(f'{e(n)} = {v["contracts"]:.2f} contracts for {_m(v["cost"])} (hits {v["if_hits"]:+.2f}, misses {v["if_misses"]:+.2f}, gives up {_m(v["ev_given_up"])})'
                                 for n, v in h["plans"].items()) + ".</p>")
        out.append(f'<details class="card"><summary><b>{e(r["label"])}</b> · stake {_m(r["stake"])} · pays {_m(r["pays"])} · {r["legs_won"]} won, {r["legs_left"]} to play · {e(state)}</summary>'
                   f'<p class="muted small" style="margin-top:6px">{e(r["note"])}</p><ul class="small">{legs}</ul>{hedge}</details>')
    return '<div class="stack">' + "".join(out) + "</div>"


def _shadow(sh: dict) -> str:
    if not sh.get("n"):
        return '<p class="muted small" style="margin-top:8px">Watch-list record: nothing wait-listed has settled yet.</p>'
    clv = f" The price for our side closed {sh['clv'] * 100:+.1f}¢ from the ask we saw on {sh['clv_n']} of them." if sh.get("clv") is not None else ""
    return (f'<p class="small" style="margin-top:8px"><b>Watch-list record</b> (as if every wait-listed bet had been bought at the ask it saw; one per player, prop and day): '
            f'{sh["n"]} positions, won {sh["won"]:.1%}, needed {sh["needed"]:.1%}, return {sh["ret"]:+.1%} per $1 (± {sh["se"]:.1%}). '
            f'{"Too few to judge." if sh["n"] < 200 or sh.get("games", 0) < 40 else ""}{clv}</p>')


def _decision(d: dict) -> str:
    t = d["thesis"]
    pill = {"ACCEPT": "good", "WAIT": "warn", "VETO": "bad", "NO BET": "warn"}[d["decision"]]
    li = lambda xs: "".join(f"<li>{e(str(x))}</li>" for x in xs) or "<li>none</li>"
    return f"""<details><summary><span class="pill {pill}">{e(d['decision'])}</span> {e(d['side'].upper())} · {e(d['title'])} <span class="muted small">ask {d['ask'] * 100:.0f}¢ · ours {d['p_est']:.1%} · EV {d['ev']:+.3f}</span></summary>
<p class="small" style="margin-top:6px"><b>Decision:</b> {e(d['reason'])}</p>
<ul><li><b>Facts</b></li>{li(t['facts'])}<li><b>Model outputs</b></li>{li(t['model_outputs'])}<li><b>Supporting</b></li>{li(t['supporting'])}<li><b>Contradicting</b></li>{li(t['contradicting'])}
<li><b>Availability</b></li>{li(t['availability'])}<li><b>Assumptions</b></li>{li(t['assumptions'])}<li><b>How it fails</b></li>{li(t['failure_scenarios'])}</ul>
<ol>{li(s.split(' ', 1)[1] if ' ' in s else s for s in t['steps'])}</ol></details>"""


def _card(c: dict) -> str:
    legs = "".join(f"<li>{e(l['side'].upper())} · {e(l['title'])} <span class='num'>{l['ask'] * 100:.0f}¢</span> · chance {l['p']:.0%} · {e(l['game_key'])}</li>" for l in c["legs"])
    pill = "bad" if c["status"] == "VETO" else "warn"
    return f"""<details><summary><span class="pill {pill}">{e(c['status'])}</span> {e(c['card_id'])} · {c['n_legs']} legs, {len(c['games'])} games · walk-away {c['need_mult']:,.1f}× <span class="muted small">({_m(c['stake'])} must pay {_m(c['max_payout'])})</span></summary>
<p class="small" style="margin-top:6px">Chance {c['p_joint']:.2%} (1 in {1 / c['p_joint']:,.0f}) · fair {c['fair_mult']:,.1f}× · expected value at walk-away {c['ev_at_need']:+.0%} · loses everything {c['p_lose']:.1%} of the time · maximum loss {_m(c['max_loss'])}.
{e(c['correlation'])}.{' Vetoed: ' + e('; '.join(c['vetoes'])) if c['vetoes'] else ''}</p><ul>{legs}</ul></details>"""


def write() -> str:
    from bot.kalshi import data_client
    from . import audit
    from .models import stack
    f = pipeline.OUT / "latest.json"
    if not f.exists():
        raise SystemExit("no run yet: python -m quantsport daily")
    o = json.loads(f.read_text())
    con = store.connect()
    try:
        perf = audit.report(data_client(), con, verbose=False)
    except Exception:
        perf = {}
    models = json.loads(stack.MODELS.read_text()) if stack.MODELS.exists() else {}
    try:
        from . import monitor
        pos = monitor.status(data_client(), con)
    except Exception:
        pos = store.frame(con, "SELECT label, tier, stake, pays, expires FROM positions WHERE status='open' ORDER BY opened DESC LIMIT 40").to_dict("records")
    out = pipeline.OUT / "dashboard.html"
    out.write_text(render(o, perf, models, pos))
    return str(out)
