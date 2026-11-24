"""The slips as one phone-sized page: tick legs off as you add them in Kalshi, see the walk-away line,
open each market. Written as a page fragment (title + style + content) so it can be published as is."""
from __future__ import annotations

from html import escape as e

import pandas as pd

CSS = """
/* layout: one narrow column of ticket stubs; the walk-away line is the loudest thing on each ticket */
:root{
  --bg:#edf0f4; --card:#ffffff; --ink:#16202b; --muted:#5b6878; --line:#d3dae3; --soft:#f4f6f9;
  --accent:#2447d6; --accent-ink:#ffffff; --good:#0f7a4d; --warn:#9a5b00; --bad:#b3261e; --hold:#fff4d6; --hold-ink:#4d3a00;
  --display:"Space Grotesk",system-ui,sans-serif; --body:"Inter",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#10151c; --card:#19212b; --ink:#e8edf3; --muted:#9aa7b6; --line:#2c3846; --soft:#141b24;
  --accent:#8aa2ff; --accent-ink:#0c1220; --good:#55c996; --warn:#e9b455; --bad:#ff8a80; --hold:#3a3012; --hold-ink:#ffe9a8; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#10151c; --card:#19212b; --ink:#e8edf3; --muted:#9aa7b6; --line:#2c3846; --soft:#141b24;
  --accent:#8aa2ff; --accent-ink:#0c1220; --good:#55c996; --warn:#e9b455; --bad:#ff8a80; --hold:#3a3012; --hold-ink:#ffe9a8; color-scheme:dark }
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.45}
.wrap{max-width:680px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:18px}
h1,h2,h3{font-family:var(--display);margin:0;text-wrap:balance}
h1{font-size:1.7rem;letter-spacing:-.01em} h2{font-size:1.15rem;margin-top:10px} h3{font-size:1.25rem}
p{margin:0} .muted{color:var(--muted)} .small{font-size:.86rem} .num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.notice{background:var(--hold);color:var(--hold-ink);border-radius:10px;padding:10px 12px;font-size:.9rem}
.sum{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.sum div{background:var(--card);padding:10px 12px;min-width:0} .sum b{display:block;font-family:var(--mono);font-size:1.05rem}
.sum span{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.slip{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.head{padding:14px 14px 12px;display:flex;flex-direction:column;gap:6px}
.tag{font-family:var(--mono);font-size:.74rem;letter-spacing:.06em;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pill{font-family:var(--body);font-size:.72rem;font-weight:600;letter-spacing:.03em;border-radius:99px;padding:2px 8px;border:1px solid currentColor}
.pill.good{color:var(--good)} .pill.warn{color:var(--warn)} .pill.bad{color:var(--bad)}
.walk{border-left:4px solid var(--accent);background:var(--soft);padding:9px 11px;font-weight:600}
.walk small{display:block;font-weight:400;color:var(--muted)}
.facts{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.86rem;color:var(--muted)} .facts b{color:var(--ink);font-family:var(--mono);font-weight:500}
.perf{height:0;border-top:2px dashed var(--line);margin:0 -1px;position:relative}
.perf::before,.perf::after{content:"";position:absolute;top:-9px;width:16px;height:16px;border-radius:50%;background:var(--bg);border:1px solid var(--line)}
.perf::before{left:-9px} .perf::after{right:-9px}
.game{padding:10px 14px 4px;font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between;gap:8px}
.leg{display:grid;grid-template-columns:28px minmax(0,1fr) auto;gap:2px 8px;align-items:start;padding:8px 14px;border-top:1px solid var(--line)}
.leg input{width:22px;height:22px;margin:1px 0 0;accent-color:var(--accent)}
.leg label{font-weight:600;min-width:0;overflow-wrap:anywhere} .leg .px{font-family:var(--mono);text-align:right}
.leg .sub{grid-column:2 / 4;font-size:.84rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:2px 12px;min-width:0}
.leg.done label{text-decoration:line-through;color:var(--muted)}
.leg details{grid-column:2 / 4;font-size:.84rem;color:var(--muted)} .leg summary{cursor:pointer;color:var(--accent)} .leg ul{margin:4px 0 2px;padding-left:18px}
a{color:var(--accent)} a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.foot{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;justify-content:space-between;padding:10px 14px;border-top:1px solid var(--line);background:var(--soft);font-size:.86rem}
button{font:600 .86rem var(--body);background:var(--accent);color:var(--accent-ink);border:0;border-radius:8px;padding:8px 12px;cursor:pointer}
.order{display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 12px;padding:0 14px 12px;font-size:.9rem}
.order dt{color:var(--muted)} .order dd{margin:0;min-width:0;overflow-wrap:anywhere}
.tick{font-family:var(--mono);font-size:.78rem;color:var(--muted);overflow-wrap:anywhere}
.expired .walk{border-left-color:var(--bad)}
@media (prefers-reduced-motion:no-preference){ .leg label{transition:color .15s} }
"""

JS = """
(function(){
  var slate=document.documentElement.querySelector('[data-slate]').dataset.slate, key='slips-'+slate, st={};
  try{ st=JSON.parse(localStorage.getItem(key)||'{}') }catch(e){}
  function save(){ try{ localStorage.setItem(key,JSON.stringify(st)) }catch(e){} }
  function count(slip){ var b=slip.querySelectorAll('.leg input'), n=0; b.forEach(function(x){ if(x.checked) n++ });
    var out=slip.querySelector('[data-count]'); if(out) out.textContent=n+' of '+b.length+' added'; }
  document.querySelectorAll('.leg input').forEach(function(x){
    if(st[x.id]){ x.checked=true; x.closest('.leg').classList.add('done') }
    x.addEventListener('change',function(){ st[x.id]=x.checked; save(); x.closest('.leg').classList.toggle('done',x.checked); count(x.closest('.slip')) });
  });
  document.querySelectorAll('.slip').forEach(count);
  document.querySelectorAll('[data-copy]').forEach(function(b){ b.addEventListener('click',function(){
    var t=document.getElementById(b.dataset.copy), text=t.textContent, done=function(){ var o=b.textContent; b.textContent='Copied'; setTimeout(function(){ b.textContent=o },1500) };
    if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(text).then(done,function(){ t.hidden=false; var r=document.createRange(); r.selectNodeContents(t); var s=getSelection(); s.removeAllRanges(); s.addRange(r) }) }
    else { t.hidden=false }
  }) });
  function clock(){ var now=Date.now(); document.querySelectorAll('[data-expires]').forEach(function(s){
    var p=s.querySelector('[data-state]'); if(!p) return;
    if(now>=Date.parse(s.dataset.expires)){ s.classList.add('expired'); p.className='pill bad'; p.textContent=s.dataset.late } }) }
  clock(); setInterval(clock,30000);
})();
"""


def _money(v: float) -> str:
    return f"${v:,.2f}" if abs(v) < 100 and v != int(v) else f"${v:,.0f}"


def _et(ts) -> str:
    from .slips import _et as f
    return f(ts)


def _core(s: dict) -> str:
    r = s.get("record") or {}
    why = ""
    if r and r["won"] == r["won"]:
        why = f"This price band since the rule switched on: {r['bets']} bets, won {r['won']:.1%}, needed {r['needs']:.1%} ({r['ret']:+.1%} per $1)."
    elif r:
        why = f"{s['rule']}: {r['bets']} bets over two seasons, {r['ret']:+.1%} per $1 after fees."
    state = ("Game is on: place now", "good") if s["live"] else (
        (f"Place before {_et(s['tipoff'])}", "good") if s.get("lo") is not None else (f"Place from tip-off, {_et(s['tipoff'])}", "warn"))
    if s["status"] == "reserve":
        state = ("Reserve: over the sleeve cap", "warn")
    return f"""
<article class="slip" id="{e(s['id'])}">
  <div class="head">
    <div class="tag"><span>{e(s['id'])} · SINGLE BET</span><span class="pill {state[1]}">{e(state[0])}</span></div>
    <h3>{e(s['pick'])}</h3>
    <p class="muted small">{e(s['matchup'])}</p>
  </div>
  <dl class="order">
    <dt>Press</dt><dd>{e(s['tap'])}</dd>
    <dt>Size</dt><dd><span class="num">{s['contracts']}</span> contracts, pay no more than <span class="num">{s['limit'] * 100:.0f}¢</span> each (order book: {s['price_seen'] * 100:.0f}¢, {s['size_shown']:.0f} offered within the limit)</dd>
    <dt>Money</dt><dd><span class="num">{_money(s['cost'])}</span> with fee. Pays <span class="num">{_money(s['pays'])}</span> if it wins (+{_money(s['profit'])}).</dd>
    <dt>Only if</dt><dd>{e(s['when'][:1].upper() + s['when'][1:])}. Outside that price range the rule was never tested.</dd>
    {f'<dt>Why</dt><dd>{e(why)}</dd>' if why else ''}
  </dl>
  <div class="foot"><a href="{e(s['url'])}" target="_blank" rel="noopener">Open this market</a><span class="tick">{e(s['ticker'])}</span></div>
</article>"""


def _ticket(t: dict) -> str:
    if t["status"] == "unbuilt":
        return (f'<article class="slip"><div class="head"><div class="tag">{e(t["id"])} · LOTTERY CARD</div>'
                f'<h3>{_money(t["stake"])} to win {_money(t["target"])}</h3><p class="muted">{e(t["problems"][0])}</p></div></article>')
    legs, g, lines = [], None, []
    for i, l in enumerate(t["legs"]):
        if l["matchup"] != g:
            g = l["matchup"]
            legs.append(f'<div class="game"><span>{e(g)}</span><span>tips {_et(l["tipoff"])}</span></div>')
        lid = f"{t['id']}-{i}"
        why = ("<details><summary>Why this leg</summary><ul>" + "".join(f"<li>{e(w)}</li>" for w in l["why"]) + "</ul></details>") if l["why"] else ""
        legs.append(f"""
  <div class="leg"><input type="checkbox" id="{lid}"><label for="{lid}">{e(l['pick'])}</label><span class="px">{l['price'] * 100:.0f}¢</span>
    <div class="sub"><span>{e(l['tap'])}</span><a href="{e(l['url'])}" target="_blank" rel="noopener">open</a></div>{why}</div>""")
        lines.append(f"{l['matchup']}: {l['tap']}")
    ok = not t["problems"]
    rule = ('<span class="pill good">Fits Kalshi\'s combo rules</span>' if ok
            else f'<span class="pill bad">{e("; ".join(t["problems"][:2]))}</span>')
    copy = f"{_money(t['stake'])} to win {_money(t['target'])} or more. Skip if the payout shown is lower.\n" + "\n".join(lines)
    return f"""
<article class="slip" id="{e(t['id'])}" data-expires="{e(t['expires'])}" data-late="A game has started: build a new card">
  <div class="head">
    <div class="tag"><span>{e(t['id'])} · LOTTERY CARD · {t['n_legs']} LEGS</span><span class="pill warn" data-state>Good until {_et(t['expires'])}</span>{rule}</div>
    <h3>{_money(t['stake'])} to win {_money(t['target'])}</h3>
    <p class="walk">Place only if Kalshi shows a payout of {_money(t['target'])} or more for {_money(t['stake'])}.
      <small>That is {t['need_x']:,.0f}× after the fee. If the payout shown is lower, skip this card. Do not add legs to chase it.</small></p>
    <div class="facts"><span>Legs worth <b>{t['fair_x']:,.0f}×</b></span><span>Hits about <b>1 in {t['one_in']:,.0f}</b></span>
      <span>Expected loss <b>{_money(t['exp_loss'])}</b></span><span><b>{t['n_games']}</b> games</span></div>
  </div>
  <div class="perf"></div>
  {''.join(legs)}
  <div class="foot"><span data-count></span><button type="button" data-copy="copy-{e(t['id'])}">Copy legs</button></div>
  <pre id="copy-{e(t['id'])}" hidden style="margin:0;padding:10px 14px;white-space:pre-wrap;font-size:.8rem">{e(copy)}</pre>
</article>"""


def render(doc: dict) -> str:
    core, tickets = doc["core"], doc["tickets"]
    spend = sum(t["stake"] for t in tickets if t["status"] == "open")
    day = pd.Timestamp(doc["slate"]).strftime("%a %b %-d")
    if core["note"]:
        core_html = f'<p class="muted">{e(core["note"])}</p>'
    elif not core["slips"]:
        core_html = (f'<p class="muted">No double-double under is priced 35–80¢ right now. {core["waiting"]} markets are outside the '
                     f'range and may enter it during play.</p>')
    else:
        core_html = "".join(_core(s) for s in core["slips"]) + (
            f'<p class="muted small">{core["waiting"]} more double-double markets are outside the range now and may enter it '
            f'during play. The live watcher logs those as they happen.</p>')
    tick_html = "".join(_ticket(t) for t in tickets) or '<p class="muted">Nothing qualifies right now. Props usually post the morning of the game.</p>'
    return f"""<title>Tonight's Slips</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">
<style>{CSS}</style>
<main class="wrap" data-slate="{e(doc['slate'])}">
  <header>
    <p class="tag">PAPER SLIPS · MADE {e(_et(doc['made_at']).upper())}</p>
    <h1>Slips for {e(day)}</h1>
  </header>
  <p class="notice">Nothing here has been placed. Prices move after this page is made, so check the price or payout Kalshi shows against the limit on each slip before you confirm.</p>
  <div class="sum">
    <div><span>Bankroll</span><b>{_money(doc['bankroll'])}</b></div>
    <div><span>Single bets</span><b>{_money(core['stake'])} each</b></div>
    <div><span>Cards tonight</span><b>{_money(spend)}</b></div>
  </div>
  <h2>Single bets</h2>
  <p class="muted small">{e(core.get('intro', ''))} At most {_money(core['sleeve'])} at risk at once.</p>
  {core_html}
  <h2>Lottery cards</h2>
  <p class="muted small">Built from unders priced 50–75¢ on tight markets. Every card has a negative expected value. Tick each leg as you add it to the combo in Kalshi.</p>
  {tick_html}
  <p class="muted small">Chances treat the legs as independent. Links open the market group on kalshi.com; if one does not, search the player's name in the app.</p>
</main>
<script>{JS}</script>
"""
