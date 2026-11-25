"""Investment mathematics for binary contracts that pay $1. Pure functions, each tested in tests/test_quantsport.py.

Prices and costs are dollars per contract (0-1). `p` is our probability the contract pays.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

TAKER, MAKER = 0.07, 0.0175          # Kalshi 'quadratic' fee: rate x price x (1 - price) per contract


def fee(price: float, rate: float = TAKER) -> float:
    return rate * price * (1 - price)


def order_fee(price: float, contracts: float, rate: float = TAKER) -> float:
    """Fee for one order: summed over contracts and rounded UP to the cent (Kalshi 'Fee Rounding', retail accounts)."""
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100 if contracts > 0 else 0.0


def ev(p: float, cost: float, fees: float = 0.0) -> float:
    """Expected profit per contract: p x (1 - cost) - (1 - p) x cost - fees."""
    return p * (1 - cost) - (1 - p) * cost - fees


def expected_return(p: float, cost: float, fees: float = 0.0) -> float:
    """Expected profit per dollar put at risk (cost plus fees)."""
    return ev(p, cost, fees) / (cost + fees)


def break_even(cost: float, fees: float = 0.0) -> float:
    """The probability at which the bet is worth exactly nothing."""
    return cost + fees


def edge(p: float, cost: float, fees: float = 0.0) -> float:
    """Our probability minus the break-even probability, in probability points."""
    return p - break_even(cost, fees)


def kelly(p: float, cost: float, fees: float = 0.0) -> float:
    """Growth-optimal share of bankroll for a $1 contract bought at cost + fees: f* = (b p - q) / b with
    b = (1 - c) / c the net profit per dollar risked. Zero when there is no edge."""
    c = cost + fees
    if not (0 < c < 1) or not (0 <= p <= 1):
        return 0.0
    b = (1 - c) / c
    return max((b * p - (1 - p)) / b, 0.0)


def kelly_sized(p: float, cost: float, fees: float = 0.0, fraction: float = 0.25, cap: float = 0.05, p_se: float = 0.0, z: float = 0.0) -> float:
    """Fractional, capped Kelly on a cautious probability (p - z x standard error). The default the fund uses."""
    return min(fraction * kelly(max(p - z * p_se, 0.0), cost, fees), cap)


def kelly_table(p: float, cost: float, fees: float = 0.0, cap: float = 0.05) -> dict:
    f = kelly(p, cost, fees)
    return {"full": f, "half": f / 2, "quarter": f / 4, "capped": min(f, cap)}


def devig(prices: list[float]) -> list[float]:
    """Remove a book's margin from the prices of mutually exclusive outcomes that should sum to 1 (multiplicative)."""
    s = sum(prices)
    return [x / s for x in prices] if s > 0 else prices


def contracts_for(stake: float, price: float, rate: float = TAKER) -> tuple[int, float]:
    """(whole contracts, total cost with the order fee) that fit inside a stake."""
    if not 0 < price < 1 or stake <= 0:
        return 0, 0.0
    n = int(stake // (price + fee(price, rate)))
    while n > 0 and n * price + order_fee(price, n, rate) > stake + 1e-9:
        n -= 1
    return n, round(n * price + order_fee(price, n, rate), 2) if n else 0.0


# ---------------------------------------------------------------- joint outcomes
def joint_independent(ps) -> float:
    return float(np.prod(ps))


def latent_rho(p1: float, p2: float, phi: float) -> float:
    """Correlation of the hidden normal variables that gives two yes/no outcomes the correlation `phi`."""
    if abs(phi) < 1e-9:
        return 0.0
    target = p1 * p2 + phi * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    a, b = stats.norm.ppf(p1), stats.norm.ppf(p2)
    f = lambda r: stats.multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, r], [r, 1]]) - target
    try:
        return float(optimize.brentq(f, -0.99, 0.99, xtol=1e-4))
    except ValueError:
        return float(np.clip(phi * 1.5, -0.95, 0.95))


def joint_copula(ps, corr, sims: int = 200_000, seed: int = 7) -> float:
    """Chance that every leg hits when the legs' hidden drivers have correlation matrix `corr` (Gaussian copula)."""
    ps = np.asarray(ps, float)
    corr = np.asarray(corr, float)
    if np.allclose(corr, np.eye(len(ps))):
        return joint_independent(ps)
    w, v = np.linalg.eigh(corr)                       # nearest valid correlation matrix
    c = (v * np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(len(ps)), c, size=sims, method="cholesky")
    return float((z < stats.norm.ppf(ps)).all(axis=1).mean())


def fair_multiple(p_joint: float) -> float:
    return 1 / p_joint if p_joint > 0 else float("inf")


def combo_ev(p_joint: float, offered_multiple: float) -> float:
    """Expected profit per $1 staked on a card that returns `offered_multiple` dollars (fee already inside it)."""
    return p_joint * offered_multiple - 1


# ---------------------------------------------------------------- risk
def drawdown_scale(drawdown: float, steps) -> float:
    """How much new risk is allowed at the current drawdown: 1.0 normally, less as losses mount, 0 at the stop."""
    s = 1.0
    for level, factor in sorted(steps):
        if drawdown >= level:
            s = factor
    return s


def var_cvar(pnl: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Loss not exceeded with probability `level`, and the average loss beyond it (both as positive dollars)."""
    losses = -np.asarray(pnl, float)
    v = float(np.quantile(losses, level))
    tail = losses[losses >= v]
    return max(v, 0.0), float(max(tail.mean(), 0.0)) if len(tail) else max(v, 0.0)
