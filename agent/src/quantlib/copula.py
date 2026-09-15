"""Copula models for non-linear dependency and tail risk analysis.

Implements bivariate Archimedean copulas (Clayton, Gumbel, Frank) and Gaussian copula,
with Kendall's tau calibration and tail dependence coefficients.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm, rankdata

__all__ = [
    "clayton_copula_cdf",
    "clayton_tail_dependence",
    "fit_copula_from_tau",
    "frank_copula_cdf",
    "gaussian_copula_cdf",
    "gumbel_copula_cdf",
    "gumbel_tail_dependence",
    "pseudo_observations",
]


def pseudo_observations(data: np.ndarray) -> np.ndarray:
    """Transform empirical data to uniform pseudo-observations via average ranks."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("data must be 1-D or 2-D")
    if np.isinf(arr).any():
        raise ValueError("data must not contain infinite values")

    if arr.ndim == 1:
        result = np.full(arr.shape, np.nan, dtype=float)
        valid = ~np.isnan(arr)
        n = int(valid.sum())
        if n:
            result[valid] = rankdata(arr[valid], method="average") / (n + 1.0)
        return result

    result = np.full(arr.shape, np.nan, dtype=float)
    for column in range(arr.shape[1]):
        valid = ~np.isnan(arr[:, column])
        n = int(valid.sum())
        if n:
            result[valid, column] = rankdata(arr[valid, column], method="average") / (n + 1.0)
    return result


def clayton_copula_cdf(u: float | np.ndarray, v: float | np.ndarray, theta: float) -> float | np.ndarray:
    if theta <= 0.0:
        raise ValueError(f"Clayton parameter theta must be strictly positive, got {theta}")
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if np.any((u_arr <= 0.0) | (u_arr > 1.0) | (v_arr <= 0.0) | (v_arr > 1.0)):
        raise ValueError("u and v marginals must be in (0, 1]")
    a = -theta * np.log(u_arr)
    b = -theta * np.log(v_arr)
    lse = np.logaddexp(a, b)
    log_inner = lse + np.log1p(-np.exp(-lse))
    val = np.exp(-log_inner / theta)
    return float(val) if np.ndim(val) == 0 else val


def clayton_tail_dependence(theta: float) -> dict[str, float]:
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    return {"lambda_lower": float(2.0 ** (-1.0 / theta)), "lambda_upper": 0.0}


def gumbel_copula_cdf(u: float | np.ndarray, v: float | np.ndarray, theta: float) -> float | np.ndarray:
    if theta < 1.0:
        raise ValueError(f"Gumbel parameter theta must be >= 1.0, got {theta}")
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if np.any((u_arr <= 0.0) | (u_arr > 1.0) | (v_arr <= 0.0) | (v_arr > 1.0)):
        raise ValueError("u and v marginals must be in (0, 1]")
    with np.errstate(divide="ignore", invalid="ignore"):
        a = theta * np.log(-np.log(u_arr))
        b = theta * np.log(-np.log(v_arr))
    val = np.exp(-np.exp(np.logaddexp(a, b) / theta))
    return float(val) if np.ndim(val) == 0 else val


def gumbel_tail_dependence(theta: float) -> dict[str, float]:
    if theta < 1.0:
        raise ValueError(f"theta must be >= 1.0, got {theta}")
    return {"lambda_lower": 0.0, "lambda_upper": float(2.0 - 2.0 ** (1.0 / theta))}


def frank_copula_cdf(u: float | np.ndarray, v: float | np.ndarray, theta: float) -> float | np.ndarray:
    if theta == 0.0:
        raise ValueError("Frank parameter theta must be non-zero (theta=0 is independence)")
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if np.any((u_arr <= 0.0) | (u_arr > 1.0) | (v_arr <= 0.0) | (v_arr > 1.0)):
        raise ValueError("u and v marginals must be in (0, 1]")
    if theta > 0:
        s = np.minimum(u_arr, v_arr)
        t = np.maximum(u_arr, v_arr)
        inner = -(np.expm1(-theta * t) + np.exp(-theta * (t - s)) * np.expm1(-theta * (1.0 - t)))
        log_num = -theta * s + np.log(inner)
        log_den = np.log(-np.expm1(-theta))
        val = -(1.0 / theta) * (log_num - log_den)
    else:
        phi = -theta
        log_num = (
            phi * u_arr + np.log(-np.expm1(-phi * u_arr))
            + phi * v_arr + np.log(-np.expm1(-phi * v_arr))
        )
        log_den = phi + np.log(-np.expm1(-phi))
        val = np.logaddexp(0.0, log_num - log_den) / phi
    return float(val) if np.ndim(val) == 0 else val


def gaussian_copula_cdf(u: float, v: float, rho: float) -> float:
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be strictly in (-1.0, 1.0), got {rho}")
    if not (0.0 < u <= 1.0 and 0.0 < v <= 1.0):
        raise ValueError("u and v must be in (0, 1]")
    from scipy.stats import multivariate_normal
    z1 = float(norm.ppf(u))
    z2 = float(norm.ppf(v))
    return float(multivariate_normal.cdf([z1, z2], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]))


def fit_copula_from_tau(tau: float, family: Literal["clayton", "gumbel", "gaussian"]) -> dict[str, float]:
    fam = family.strip().lower()
    if fam == "clayton":
        if tau <= 0.0 or tau >= 1.0:
            raise ValueError("Clayton copula requires tau in (0, 1)")
        theta = float(2.0 * tau / (1.0 - tau))
        return {"family": "clayton", "theta": theta, "tau": tau, **clayton_tail_dependence(theta)}
    if fam == "gumbel":
        if tau < 0.0 or tau >= 1.0:
            raise ValueError("Gumbel copula requires tau in [0, 1)")
        theta = float(1.0 / (1.0 - tau))
        return {"family": "gumbel", "theta": theta, "tau": tau, **gumbel_tail_dependence(theta)}
    if fam == "gaussian":
        if not (-1.0 < tau < 1.0):
            raise ValueError("Gaussian copula requires tau in (-1, 1)")
        rho = float(math.sin(math.pi * 0.5 * tau))
        return {"family": "gaussian", "rho": rho, "tau": tau, "lambda_lower": 0.0, "lambda_upper": 0.0}
    raise ValueError(f"Unsupported family: {family}")
