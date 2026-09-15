"""Market-impact and slippage models for backtest execution.

Four models, ordered by how much of the order book they claim to know:

========================= ================================== =====================
Model                     Impact                             Use when
========================= ================================== =====================
:func:`fixed_slippage`    constant ``bps``                   order < 0.5% of ADV
:func:`linear_impact`     ``coeff * V / ADV``                order 0.5-5% of ADV
:func:`sqrt_impact`       ``eta * sigma * sqrt(V / ADV)``    order > 5% of ADV
:func:`delayed_execution` shifts a signal forward in time    always, for signal lag
========================= ================================== =====================

All three price models compute a non-negative impact and push the fill price
*against* the trader -- up when buying, down when selling.

The two size-aware models, :func:`linear_impact` and :func:`sqrt_impact`, are
additionally monotone non-decreasing in order size and return the untouched price
at zero size. :func:`fixed_slippage` takes no size at all: it charges its ``bps``
on every fill regardless of how small, which is the whole point of a fixed model
and the reason it is only appropriate below roughly 0.5% of ADV.

Every function here is scalar-only. Passing a numpy array or a pandas Series where
a ``float`` is documented raises rather than broadcasting.

These are backtest primitives. Nothing here places, routes or prices a live order.

Note on naming: the square-root model is frequently labelled "Almgren-Chriss" and
is indeed the impact term from that literature, but this module does **not**
implement Almgren-Chriss optimal execution -- there is no trading trajectory, no
permanent/temporary impact split and no risk-aversion parameter. Call it what it
is: a square-root impact function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "DEFAULT_DELAY_BARS",
    "DEFAULT_LINEAR_IMPACT_COEFF",
    "DEFAULT_SLIPPAGE_BPS",
    "DEFAULT_SQRT_IMPACT_ETA",
    "delayed_execution",
    "fixed_slippage",
    "linear_impact",
    "sqrt_impact",
]

DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_LINEAR_IMPACT_COEFF = 0.1
DEFAULT_SQRT_IMPACT_ETA = 0.5
DEFAULT_DELAY_BARS = 1
_BPS_PER_UNIT = 10_000.0


def _check_order(price: ArrayLike, direction: int) -> np.ndarray:
    prices = np.asarray(price, dtype=float)
    if not np.all(np.isfinite(prices) & (prices > 0.0)):
        raise ValueError(f"price must be finite and strictly positive, got {price!r}")
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 (buy) or -1 (sell), got {direction!r}")
    return prices


def _participation_rate(volume_traded: ArrayLike, adv: ArrayLike) -> np.ndarray:
    volumes = np.asarray(volume_traded, dtype=float)
    advs = np.asarray(adv, dtype=float)
    if not np.all(np.isfinite(volumes) & (volumes >= 0.0)):
        raise ValueError(f"volume_traded must be finite and non-negative, got {volume_traded!r}")
    if not np.all(np.isfinite(advs) & (advs > 0.0)):
        raise ValueError(f"adv must be finite and strictly positive, got {adv!r}")
    return volumes / advs


def _fill(prices: np.ndarray, direction: int, impact: np.ndarray) -> float | np.ndarray:
    fill = prices * (1.0 + direction * impact)
    if not np.all(np.isfinite(fill) & (fill > 0.0)):
        raise ValueError("impact produces a non-finite or non-positive fill price")
    return float(fill) if fill.ndim == 0 else fill


def fixed_slippage(price: float, direction: int, bps: float = DEFAULT_SLIPPAGE_BPS) -> float:
    prices = _check_order(price, direction)
    rates = np.asarray(bps, dtype=float)
    if not np.all(np.isfinite(rates) & (rates >= 0.0)):
        raise ValueError(f"bps must be finite and non-negative, got {bps!r}")
    return _fill(prices, direction, rates / _BPS_PER_UNIT)


def linear_impact(
    price: float,
    direction: int,
    volume_traded: float,
    adv: float,
    impact_coeff: float = DEFAULT_LINEAR_IMPACT_COEFF,
) -> float:
    prices = _check_order(price, direction)
    coeffs = np.asarray(impact_coeff, dtype=float)
    if not np.all(np.isfinite(coeffs) & (coeffs >= 0.0)):
        raise ValueError(f"impact_coeff must be finite and non-negative, got {impact_coeff!r}")
    return _fill(prices, direction, coeffs * _participation_rate(volume_traded, adv))


def sqrt_impact(
    price: float,
    direction: int,
    volume_traded: float,
    adv: float,
    volatility: float,
    eta: float = DEFAULT_SQRT_IMPACT_ETA,
) -> float:
    prices = _check_order(price, direction)
    vols = np.asarray(volatility, dtype=float)
    etas = np.asarray(eta, dtype=float)
    if not np.all(np.isfinite(vols) & (vols >= 0.0)):
        raise ValueError(f"volatility must be finite and non-negative, got {volatility!r}")
    if not np.all(np.isfinite(etas) & (etas >= 0.0)):
        raise ValueError(f"eta must be finite and non-negative, got {eta!r}")
    impact = etas * vols * np.sqrt(_participation_rate(volume_traded, adv))
    return _fill(prices, direction, impact)


def delayed_execution(signal_series: pd.Series, delay_bars: int = DEFAULT_DELAY_BARS) -> pd.Series:
    if not isinstance(signal_series, pd.Series):
        raise TypeError(f"signal_series must be a pandas Series, got {type(signal_series).__name__}")
    if delay_bars < 0:
        raise ValueError(f"delay_bars must be non-negative; a negative shift is look-ahead bias, got {delay_bars!r}")
    return signal_series.shift(delay_bars)
