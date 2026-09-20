"""确定性特殊函数：正则化不完全贝塔函数及 F/t 分布。

纯 Python 实现（连分式法，源自 Numerical Recipes 的 betacf），不依赖 scipy，
保证跨平台、跨版本结果一致。所有函数均为固定迭代上限的确定性算法。
"""
from __future__ import annotations

import math

_FPMIN = 1e-300
_EPS = 3e-16
_MAX_ITER = 200


def _betacf(a: float, b: float, x: float) -> float:
    """不完全贝塔函数的连分式部分（固定迭代上限）。"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_sf(f: float, d1: float, d2: float) -> float:
    """F 分布右尾概率 P(F > f)，自由度 (d1, d2)。"""
    if f <= 0.0:
        return 1.0
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def _t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    ib = betai(df / 2.0, 0.5, x)
    if t >= 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """t 分布逆累积分布函数（二分法，固定 200 次迭代，结果确定）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在开区间 (0, 1) 内")
    if df < 1.0:
        raise ValueError("自由度必须 >= 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
