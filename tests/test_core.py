import base64
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from bot.kalshi import Kalshi, fee
from bot.markets import parse_market
from bot import combos
from bot.model import Model, build_priors, market_scores


def test_signature_matches_kalshi_spec(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "k.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    k = Kalshi("https://demo-api.kalshi.co/trade-api/v2", "kid", str(pem))
    h = k._auth_headers("get", "/portfolio/balance")
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/balance").encode()
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg,
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())
    assert h["KALSHI-ACCESS-KEY"] == "kid"


def test_fee():
    assert fee(0.5, 100) == pytest.approx(1.75)


def _m(ticker, title, **kw):
    start = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    d = dict(ticker=ticker, event_ticker="-".join(ticker.split("-")[:2]), title=title, status="active",
             yes_bid_dollars="0.40", yes_ask_dollars="0.42", occurrence_datetime=start, strike_type="greater")
    d.update(kw)
    return d


def test_parse_prop_and_game_legs():
    p = parse_market("KXNBAPTS", _m("KXNBAPTS-26NOV22OKCMIA-OKCSGILGEOUS2-30", "Shai Gilgeous-Alexander: 30+ points",
                                    floor_strike=29.5))
    assert (p.kind, p.player, p.team, p.stats, p.line, p.away, p.home) == \
        ("prop", "Shai Gilgeous-Alexander", "OKC", ("pts",), 29.5, "OKC", "MIA")
    s = parse_market("KXNBASPREAD", _m("KXNBASPREAD-26NOV22OKCMIA-OKC7", "OKC wins by over 6.5", floor_strike=6.5))
    assert (s.kind, s.team, s.line) == ("spread", "OKC", 6.5)
    w = parse_market("KXNBAGAME", _m("KXNBAGAME-26NOV22OKCMIA-MIA", "Miami wins"))
    assert (w.kind, w.team) == ("win", "MIA")


@pytest.fixture(scope="module")
def model():
    from bot.data import load_logs
    logs = load_logs(topup=False)
    return Model(logs, priors=build_priors(logs))


def test_same_game_correlation_is_positive(model):
    """Star scoring big and the game going over should be positively correlated."""
    g = model.logs[model.logs.season == model.logs.season.max()]
    top = g.groupby(["player_name", "team"]).pts.mean().sort_values().index[-1]
    name, team = top
    team = {"SAN": "SAS"}.get(team, team)
    opp = "MIA" if team != "MIA" else "BOS"
    mk = lambda t, title, **kw: parse_market(t.split("-")[0], _m(t, title, **kw))
    proj = model.project(name, team, opp)
    line = float(np.floor(proj.mean["pts"])) + 0.5  # near the median, where correlation shows most
    star = mk(f"KXNBAPTS-26NOV22{opp}{team}-{team}X-{int(line + .5)}", f"{name}: {int(line + .5)}+ points",
              floor_strike=line)
    star.player, star.team = name, team
    over = mk(f"KXNBATOTAL-26NOV22{opp}{team}-230", "Over 229.5", floor_strike=229.5)
    sim = model.simulate_game(opp, team, [(name, team)], market=market_scores([over]))
    gs = combos.GameSims([star, over], sim)
    for l in (star, over):
        l.p_fair = l.p_model
    joint = gs.joint([star.ticker, over.ticker], [star.p_fair, over.p_fair])
    assert joint > star.p_fair * over.p_fair * 1.03


def test_kelly_stake_zero_without_edge():
    c = combos.Combo(legs=[], p_fair=0.05, p_mid=0.05, est_price=0.055, corr_lift=1)
    assert c.ev < 0 and c.stake() == 0


def test_live_games_excluded():
    from bot.markets import fetch_legs

    class K:
        def markets(self, s):
            if s != "KXNBAGAME":
                return []
            soon_end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()   # tipped ~2h ago
            later = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()
            return [_m("KXNBAGAME-26NOV22OKCMIA-MIA", "Miami wins", occurrence_datetime=soon_end),
                    _m("KXNBAGAME-26NOV23PORPHI-PHI", "Philadelphia wins", occurrence_datetime=later)]
    legs = fetch_legs(K(), series=("KXNBAGAME",))
    assert [l.ticker for l in legs] == ["KXNBAGAME-26NOV23PORPHI-PHI"]


def test_fund_env_follows_gymnasium_api_and_cash_is_flat():
    import pandas as pd
    from gymnasium.utils.env_checker import check_env
    from bot.sim_env import FundEnv, cash, evaluate
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    games = pd.DataFrame([dict(date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), pool="x", game=f"g{d}{i}",
                               created_time=t0 + pd.Timedelta(days=d, minutes=10 * i),
                               resolve=t0 + pd.Timedelta(days=d, minutes=10 * i + 30), cost=0.95, win=float(i != 0 or d % 4))
                          for d in range(12) for i in range(4)])
    check_env(FundEnv(games, seed=1), skip_render_check=True)
    r = evaluate(dict(games=games, episode_nights=10), cash, episodes=5)
    assert r["median_end"] == 100 and r["bets"] == 0


def test_kelly_is_zero_at_or_below_break_even():
    from bot.fund import cost_with_fee, kelly
    assert kelly(cost_with_fee(0.95), 0.95) == 0 and kelly(0.94, 0.95) == 0 and kelly(0.99, 0.95) > 0.5


def test_team_styles_use_only_earlier_games():
    """Style metrics for a game must not change when that game's own box score changes."""
    from bot.data import load_logs
    from bot import styles as S
    logs = load_logs(topup=False)
    logs = logs[logs.season == logs.season.max()]
    a = S.team_styles(logs)
    gid = a.dropna(subset=["pace"]).game_id.iloc[-1]
    bumped = logs.copy()
    bumped.loc[bumped.game_id == gid, ["pts", "fga", "fg3a", "reb", "ast"]] *= 3
    b = S.team_styles(bumped)
    cols = ["pace", "off_rtg", "def_rtg", "three_rate", "d_three_rate"]
    pd_a = a[a.game_id == gid][cols].values
    pd_b = b[b.game_id == gid][cols].values
    assert np.allclose(pd_a, pd_b, equal_nan=True)
