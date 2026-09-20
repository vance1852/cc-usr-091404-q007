"""批次结论的组合策略：把全部有效测定合并为最终效价。

策略在方法版本上预先配置，结论生成时系统纳入所有有效且未被取代的测定，
不提供任何挑选入口 —— 从机制上防止"只挑有利结果"。
"""
from __future__ import annotations

import math

from .special import t_ppf

STRATEGIES = ("mean", "inverse_variance", "median")


def combine_assay(items: list[dict], strategy: str) -> dict:
    """合并有效测定。

    items: [{"log_potency": float, "se": float, "df": int}, ...]（log10 效价尺度）
    返回合并效价（%）、95% 置信区间及纳入的测定数。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知组合策略: {strategy}")
    if not items:
        raise ValueError("没有可合并的有效测定")
    logs = [float(it["log_potency"]) for it in items]
    ses = [float(it["se"]) for it in items]
    dfs = [int(it["df"]) for it in items]
    n = len(logs)

    se_combined: float | None
    df_combined: int | None
    if strategy == "mean":
        combined = sum(logs) / n
        if n > 1:
            var = sum((v - combined) ** 2 for v in logs) / (n - 1)
            se_combined = math.sqrt(var / n)
            df_combined = n - 1
        else:
            se_combined = ses[0]
            df_combined = dfs[0]
    elif strategy == "inverse_variance":
        weights = [1.0 / (s * s) for s in ses]
        w_sum = sum(weights)
        combined = sum(w * v for w, v in zip(weights, logs)) / w_sum
        se_combined = math.sqrt(1.0 / w_sum)
        df_combined = (n - 1) if n > 1 else dfs[0]
    else:  # median
        ordered = sorted(logs)
        mid = n // 2
        combined = ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
        se_combined = None
        df_combined = None

    potency_pct = (10.0 ** combined) * 100.0
    ci_pct = None
    if se_combined is not None and df_combined is not None and df_combined >= 1:
        tcrit = t_ppf(0.975, float(df_combined))
        lo, hi = combined - tcrit * se_combined, combined + tcrit * se_combined
        ci_pct = [(10.0 ** lo) * 100.0, (10.0 ** hi) * 100.0]
    return {
        "strategy": strategy,
        "n_determinations": n,
        "log_potency": combined,
        "se_log_potency": se_combined,
        "df": df_combined,
        "potency_pct": potency_pct,
        "ci95_pct": ci_pct,
    }
