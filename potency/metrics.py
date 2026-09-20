"""效价计算指标：复孔精密度、有效范围、平行性检验、相对效价。

所有指标均为输入数据的确定性函数。
"""
from __future__ import annotations

import math

from .fitting import CurveFit, fit_4pl, fit_shared_asymptotes
from .special import f_sf, t_ppf


def group_by_level(levels, responses) -> dict[int, list[float]]:
    """按稀释水平分组（键升序，保证确定性迭代顺序）。"""
    groups: dict[int, list[float]] = {}
    for lvl, y in zip(levels, responses):
        groups.setdefault(int(lvl), []).append(float(y))
    return dict(sorted(groups.items()))


def replicate_rsd(groups: dict[int, list[float]]) -> dict:
    """各稀释水平复孔的相对标准偏差（%），并给出最大值。"""
    per_level: dict[int, float | None] = {}
    for lvl, ys in groups.items():
        if len(ys) >= 2:
            mean = sum(ys) / len(ys)
            var = sum((v - mean) ** 2 for v in ys) / (len(ys) - 1)
            per_level[lvl] = (math.sqrt(var) / abs(mean) * 100.0) if mean != 0.0 else None
        else:
            per_level[lvl] = 0.0
    finite = [v for v in per_level.values() if v is not None]
    return {
        "per_level": {str(k): v for k, v in per_level.items()},
        "max_rsd_pct": max(finite) if finite else None,
    }


def valid_levels(
    groups: dict[int, list[float]],
    A: float,
    D: float,
    low_frac: float,
    high_frac: float,
) -> list[int]:
    """有效范围：水平均值落在 [D+low*(A-D), D+high*(A-D)] 窗口内的稀释水平。

    即响应处于上下平台之间、远离平台饱和区的稀释度（默认 10%~90% 跨度）。
    """
    lo = D + low_frac * (A - D)
    hi = D + high_frac * (A - D)
    ymin, ymax = min(lo, hi), max(lo, hi)
    out = []
    for lvl, ys in groups.items():
        mean = sum(ys) / len(ys)
        if ymin <= mean <= ymax:
            out.append(lvl)
    return out


def parallelism_test(
    x_std,
    y_std,
    x_smp,
    y_smp,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> dict:
    """平行性 F 检验：比较"各自独立拟合"与"共享 A/B/D 仅平移 logC"两个模型。

    F = ((RSS_约束 - RSS_独立) / 3) / (RSS_独立 / (n - 8))
    p 值越小表示两条曲线越不平行。返回原始统计量，判定阈值由规则集给出。
    """
    f_std = fit_4pl(x_std, y_std, max_iter=max_iter, tol=tol)
    f_smp = fit_4pl(x_smp, y_smp, max_iter=max_iter, tol=tol)
    shared = fit_shared_asymptotes(
        [(x_std, y_std), (x_smp, y_smp)], max_iter=max_iter, tol=tol
    )
    result = {
        "test": "parallelism_f_test",
        "F": None,
        "df1": None,
        "df2": None,
        "p": None,
        "rss_unconstrained": None,
        "rss_constrained": None,
        "note": "",
    }
    if not (f_std.converged and f_smp.converged and shared.status == "CONVERGED"):
        result["note"] = "曲线拟合未收敛，无法检验平行性"
        return result
    n = f_std.n_points + f_smp.n_points
    df1 = 3  # 独立模型 8 参数 vs 约束模型 5 参数
    df2 = n - 8
    rss_u = f_std.rss + f_smp.rss
    rss_c = shared.rss
    result["rss_unconstrained"] = rss_u
    result["rss_constrained"] = rss_c
    if df2 < 1:
        result["note"] = "数据点不足，无法检验平行性"
        return result
    if rss_u <= 0.0:
        F = 0.0 if rss_c <= 0.0 else float("inf")
    else:
        F = ((rss_c - rss_u) / df1) / (rss_u / df2)
        F = max(F, 0.0)  # 数值误差可能产生微小负值
    result["F"] = F
    result["df1"] = df1
    result["df2"] = df2
    result["p"] = f_sf(F, df1, df2) if math.isfinite(F) else 0.0
    return result


def _safe_pow10(x: float | None) -> float | None:
    """10**x 的安全版本：溢出或非有限输入返回 None。"""
    if x is None or not math.isfinite(x):
        return None
    try:
        v = 10.0 ** x
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


def relative_potency(f_std: CurveFit, f_smp: CurveFit) -> dict:
    """相对效价：两曲线 EC50 的水平位移（假定已平行）。

    potency% = 10^(logC_std - logC_smp) * 100
    置信区间由两条曲线 logC 标准误合成（t 分布，双侧 95%）。
    """
    dlog = f_std.logC - f_smp.logC
    df = f_std.n_points + f_smp.n_points - 8
    se = None
    if f_std.se_logC is not None and f_smp.se_logC is not None:
        se = math.hypot(f_std.se_logC, f_smp.se_logC)
    pow10 = _safe_pow10(dlog)
    potency_pct = pow10 * 100.0 if pow10 is not None else None
    estimable = (
        math.isfinite(dlog) and se is not None and math.isfinite(se)
        and potency_pct is not None
    )
    out = {
        "log_potency": dlog if math.isfinite(dlog) else None,
        "potency_pct": potency_pct,
        "se_log_potency": se,
        "df": df,
        "ci95_log": None,
        "ci95_pct": None,
        "estimable": bool(estimable),
    }
    if estimable and df >= 1:
        tcrit = t_ppf(0.975, float(df))
        lo, hi = dlog - tcrit * se, dlog + tcrit * se
        lo10, hi10 = _safe_pow10(lo), _safe_pow10(hi)
        out["ci95_log"] = [lo, hi]
        out["ci95_pct"] = (
            [lo10 * 100.0, hi10 * 100.0]
            if lo10 is not None and hi10 is not None else None
        )
    return out
