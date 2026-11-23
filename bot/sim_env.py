"""FundEnv: a gymnasium environment that replays REAL Kalshi game nights.

One step = one NBA game night taken from history (last season's archive and this season). The
night's opportunities are the actual first-touch entries of a rule (entry time, price paid, whether
it won), processed in the order they happened, with money locked up until each game ends.

  observation  [bankroll / start, drawdown from peak, games tonight / 15,
                rule's win rate seen so far, break-even rate, standard error, share of episode left]
  action       [stake per bet as a share of bankroll (0..0.5),
                max bets tonight / 15 (0..1),
                lottery spend tonight as a share of bankroll (0..0.1)]
  reward       log(bankroll after / bankroll before)   -> the sum over an episode is log growth,
                                                          which is what Kelly sizing maximises

Episodes either bootstrap nights (so a policy cannot memorise the calendar) or replay them in
order. Policies are plain functions of the observation; `evaluate` scores them over many episodes.
The environment follows the gymnasium API, so an RL library can be plugged in later - but with a
few hundred nights and a 3-number action, direct comparison of simple policies is the right tool.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd

from . import config
from .fund import cost_with_fee, kelly


def load_nights(threshold: float = 0.95, league: str = "NBA", series: str = "KXNBAGAME", rule: str = "") -> pd.DataFrame:
    """One row per bet: slate date, pool (archive / this season), entry and resolve times, cost, win.
    rule='dd_under' replays the double-double under rule instead of a moneyline first-touch."""
    from .favorites import slate_date
    from .trades import PRIOR_TOUCH
    if rule == "dd_under":
        from .fund import dd_under_bets
        d = dd_under_bets()
        d["pool"] = d.season.map({"last season": "archive", "this season": "this season"})
        d["game"] = d.ticker                                  # each market is its own bet
        return d[["date", "pool", "game", "created_time", "resolve", "cost", "win"]].dropna(subset=["date"]).sort_values(["date", "created_time"])
    out = []
    for pool, path in (("archive", PRIOR_TOUCH), ("this season", config.DATA_DIR / "first_touch.parquet")):
        if not path.exists():
            continue
        ft = pd.read_parquet(path)
        d = ft[(ft.league == league) & (ft.series == series) & (ft.threshold == threshold)]
        g = d.sort_values("created_time").groupby("game").first().reset_index()
        g["date"] = g.game.map(slate_date)
        g["resolve"] = g.created_time + pd.to_timedelta((150 - g.mins).clip(lower=5), unit="m")
        g["pool"] = pool
        out.append(g[["date", "pool", "game", "created_time", "resolve", "cost", "win"]])
    return pd.concat(out, ignore_index=True).dropna(subset=["date"]).sort_values(["date", "created_time"])


class FundEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, games: pd.DataFrame, bankroll: float = 100.0, episode_nights: int = 60,
                 order: str = "bootstrap", max_concurrent: float = 0.5, tickets=((5.0, 1000.0),),
                 ticket_keep: float = 0.5, prior_bets: float = 0.0, seed: int | None = None,
                 max_stake: float = float("inf")):
        self.max_stake = max_stake           # dollars that can realistically be filled per bet (liquidity)
        self.nights = [g for _, g in games.groupby("date")]
        self.start, self.T, self.order = bankroll, episode_nights, order
        self.max_concurrent, self.tickets, self.keep, self.prior = max_concurrent, tickets, ticket_keep, prior_bets
        self.observation_space = gym.spaces.Box(0.0, np.inf, (7,), np.float64)
        self.action_space = gym.spaces.Box(np.zeros(3), np.array([0.5, 1.0, 0.1]), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n = len(self.nights)
        if self.order == "chronological":
            self.idx = list(range(n))
            self.T = n
        else:
            self.idx = list(self._rng.integers(0, n, self.T))
        self.t, self.bal, self.peak = 0, self.start, self.start
        self.seen_n = self.seen_w = self.seen_cost = 0.0
        return self._obs(), {}

    def _obs(self):
        n = self.seen_n
        p = self.seen_w / n if n else 0.0
        c = cost_with_fee(self.seen_cost / n) if n else 0.0
        se = math.sqrt(max(p * (1 - p), 1e-9) / n) if n else 1.0
        g = len(self.nights[self.idx[self.t]]) if self.t < len(self.idx) else 0
        return np.array([self.bal / self.start, (self.peak - self.bal) / max(self.peak, 1e-9), g / 15, p, c, se,
                         1 - self.t / max(len(self.idx), 1)], dtype=np.float64)

    def step(self, action):
        stake_frac, max_bets, ticket_frac = (float(x) for x in np.clip(action, self.action_space.low, self.action_space.high))
        night = self.nights[self.idx[self.t]]
        before, cash, open_pos, bets = self.bal, self.bal, [], 0
        max_bets = int(round(max_bets * 15))
        for r in night.itertuples():
            for pos in [p for p in open_pos if p[0] <= r.created_time]:     # games that have ended pay out
                cash += pos[1]
                open_pos.remove(pos)
            exposure = sum(p[2] for p in open_pos)
            stake = min(stake_frac * self.bal, cash, max(self.max_concurrent * self.bal - exposure, 0.0), self.max_stake)
            if stake > 0.01 and bets < max_bets:
                cash -= stake
                c = cost_with_fee(r.cost)
                open_pos.append((r.resolve, stake / c if r.win else 0.0, stake))
                bets += 1
            self.seen_n += 1                                              # every trigger is observed, bet or not
            self.seen_w += r.win
            self.seen_cost += r.cost
        cash += sum(p[1] for p in open_pos)
        spend = min(ticket_frac * before, cash)
        if spend > 0 and self.tickets:
            per = spend / len(self.tickets)
            cash -= spend
            for st, goal in self.tickets:
                mult = goal / st
                if self._rng.random() < self.keep / mult:
                    cash += per * mult
        self.bal = max(cash, 1e-6)
        self.peak = max(self.peak, self.bal)
        self.t += 1
        done = self.t >= len(self.idx)
        return self._obs(), math.log(self.bal / before), done, self.bal < 1.0, {"bets": bets, "bankroll": self.bal}


# ---------------------------------------------------------------------- policies
def cash(obs):
    return np.array([0.0, 0.0, 0.0])


def one_big_bet(obs):            # the whole 50% core sleeve on the night's first trigger
    return np.array([0.5, 1 / 15, 0.0])


def fixed(frac, tickets=0.0):
    return lambda obs: np.array([frac, 1.0, tickets])


def evidence_kelly(z: float = 1.64, prior_bets: float = 200.0, cap: float = 0.25, tickets_from_profit: float = 0.0):
    """Stake the Kelly fraction of the win rate's LOWER bound, where the win rate is what this fund has
    actually observed, blended with a sceptical prior worth `prior_bets` bets at exactly break-even.
    It bets nothing until the evidence clears that bar, and stops when it no longer does."""
    def act(obs):
        p, c, se = obs[3], obs[4], obs[5]
        n = (p * (1 - p) / se ** 2) if se < 1 and p > 0 else 0.0
        if n < 1:
            return np.array([0.0, 1.0, 0.0])
        post = (p * n + c * prior_bets) / (n + prior_bets)
        lb = post - z * math.sqrt(max(post * (1 - post), 1e-9) / (n + prior_bets))
        price = c / 1.0035 if c else 0.0
        f = min(max((lb - c) / max(1 - c, 1e-9), 0.0), cap)
        t = tickets_from_profit if obs[0] > 1.0 + tickets_from_profit else 0.0
        return np.array([f, 1.0, t])
    return act


def evaluate(env_kwargs: dict, policy, episodes: int = 500, seed: int = 0) -> dict:
    env = FundEnv(seed=seed, **env_kwargs)
    finals, growth, dd, bets = [], [], [], []
    for ep in range(episodes if env.order != "chronological" else 1):
        obs, _ = env.reset()
        tot, worst, nb, done = 0.0, 0.0, 0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            tot += r
            worst = max(worst, obs[1])
            nb += info["bets"]
            done = term or trunc
        finals.append(env.bal); growth.append(tot / max(env.t, 1)); dd.append(worst); bets.append(nb)
    f = np.array(finals)
    return dict(median_end=float(np.median(f)), bad_1in10=float(np.percentile(f, 10)), good_1in10=float(np.percentile(f, 90)),
                ahead=float((f > env.start).mean()), halved=float((f < env.start / 2).mean()),
                growth_per_night=float(np.mean(growth)), worst_drawdown=float(np.median(dd)), bets=float(np.mean(bets)))


POLICIES = {
    "stay in cash": cash,
    "50% on one bet a night (your example)": one_big_bet,
    "20% per trigger (Kelly on this season's low-end estimate)": fixed(0.195),
    "6% per trigger (spread plan)": fixed(0.0611),
    "2% per trigger": fixed(0.02),
    "evidence-based Kelly (bets only once its own record clears break-even)": evidence_kelly(),
    "6% per trigger + $10 of tickets nightly": fixed(0.0611, 0.10),
}


def run(threshold: float = 0.95, episodes: int = 400, nights: int = 60, rule: str = "", max_stake: float = float("inf")):
    pd.set_option("display.width", 230)
    games = load_nights(threshold, rule=rule)
    name = "double-double UNDER, first in-game fill at 50-80c" if rule == "dd_under" else f"moneyline, first in-game touch of {threshold:.2f}"
    print(f"\nFundEnv: {games.game.nunique():,} real bets on {games.date.nunique()} nights "
          f"({(games.pool == 'archive').sum():,} last season, {(games.pool == 'this season').sum():,} this season); "
          f"rule = {name}.  Start $100.  "
          + ("Assumes any stake fills at the tested price." if max_stake == float("inf")
             else f"No more than ${max_stake:g} can be filled per bet (liquidity cap)."))
    for label, g, order in (("THIS SEASON'S NIGHTS, reshuffled", games[games.pool == "this season"], "bootstrap"),
                            ("LAST SEASON'S NIGHTS (archive), reshuffled", games[games.pool == "archive"], "bootstrap"),
                            ("EVERYTHING IN ORDER, as it actually happened (one path)", games, "chronological")):
        rows = []
        for name, pol in POLICIES.items():
            r = evaluate(dict(games=g, episode_nights=nights, order=order, max_stake=max_stake), pol, episodes)
            rows.append(dict(policy=name, median_end=f"${r['median_end']:,.0f}", bad_1in10=f"${r['bad_1in10']:,.0f}",
                             good_1in10=f"${r['good_1in10']:,.0f}", ahead=f"{r['ahead']:.0%}", halved=f"{r['halved']:.0%}",
                             growth_night=f"{r['growth_per_night']:+.2%}", bets=f"{r['bets']:.0f}"))
        n_n = g.date.nunique() if order == "chronological" else nights
        print(f"\n{label}  ({n_n} nights per run" + ("" if order == "chronological" else f", {episodes} runs") + ")")
        print(pd.DataFrame(rows).to_string(index=False))
