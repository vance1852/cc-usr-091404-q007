"""规则集评估。

规则集是带生效日期的不可变版本：判定某块板是否有效时，**只使用该板
首轮结论锁定当时生效的规则集**。历史规则永久保留，保证旧结论可按旧
规则复算。

每条规则是一个纯函数：``check(fit_result, context) -> RuleHit | None``。
命中即记录规则代码、严重级别、实际值与限值，不做布尔掩盖——会审页面
需要展示全部命中项。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RuleHit:
    code: str
    severity: str  # "invalid"（判废） | "warning"
    message: str
    observed: Optional[float] = None
    limit: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass
class RuleSet:
    code: str
    version: str
    effective_from: date
    description: str
    checks: list = field(default_factory=list)  # list[RuleCheck]
    # 放行限（相对于标示效价的百分比区间），例如 (80, 125)
    release_low_pct: float = 80.0
    release_high_pct: float = 125.0

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "version": self.version,
            "effective_from": self.effective_from.isoformat(),
            "description": self.description,
            "release_range_pct": [self.release_low_pct, self.release_high_pct],
            "rules": [c.code for c in self.checks],
        }


@dataclass
class RuleCheck:
    code: str
    severity: str
    message: str
    fn: Callable[[dict, dict], Optional[tuple]]  # 返回 (observed, limit) 或 None


# ---------------------------------------------------------------------------
# 规则实现
# ---------------------------------------------------------------------------


def _std(fit: dict) -> dict:
    return fit["standard"]


def _smp(fit: dict) -> dict:
    return fit["sample"]


def _rng(v):
    return v if isinstance(v, (tuple, list)) else (v, v)


def check_convergence(fit, ctx):
    if not fit["converged"]:
        return (0.0, 1.0)
    return None


def check_parallelism(fit, ctx):
    p = fit["parallelism"]
    if not p["passed"]:
        return (p["p_value"], ctx.get("parallel_p_min", 0.05))
    return None


def check_potency_ci_width(fit, ctx):
    """效价 95% 置信区间半宽占估计值百分比（精密度/可靠性）。"""
    limit = ctx["ci_half_width_max_pct"]
    hw = fit["ci_half_width_percent"]
    if hw is None or (isinstance(hw, float) and math.isnan(hw)):
        return (None, limit)
    return (hw, limit) if hw > limit else None


def _make_dose_cv_check(role):
    def check(fit, ctx):
        limit = ctx["max_replicate_cv_pct"]
        curve = fit[role]
        mx = curve["max_cv_percent"]
        if mx is None:
            return None
        return (mx, limit) if mx > limit else None

    check.__name__ = f"check_{role}_cv"
    return check


def _make_recovery_check(role):
    def check(fit, ctx):
        lo, hi = ctx["recovery_pct_range"]
        curve = fit[role]
        mn, mx = curve["min_recovery_percent"], curve["max_recovery_percent"]
        if mn is None:
            return None
        bad = mn < lo or mx > hi
        # 仅统计落在有效剂量范围内的点（dose_stats 本身即定量范围）
        worst = mn if abs(mn - 100) >= abs(mx - 100) else mx
        return (worst, [lo, hi]) if bad else None

    check.__name__ = f"check_{role}_recovery"
    return check


def check_quantitative_range(fit, ctx):
    """样品 EC50 必须落在标准品实测剂量范围内（有效范围）。"""
    std = _std(fit)
    smp_ec50 = fit["sample"]["ec50"]
    lo, hi = std["quantitative_range"]
    if lo is None:
        return (smp_ec50, None)
    return (smp_ec50, [lo, hi]) if not (lo <= smp_ec50 <= hi) else None


def check_min_replicates(fit, ctx):
    need = ctx["min_replicates_per_dose"]
    for role in ("standard", "sample"):
        for ds in fit[role]["dose_stats"]:
            if ds["n"] < need:
                return (ds["n"], need)
    return None


def check_min_dose_levels(fit, ctx):
    need = ctx["min_dose_levels"]
    for role in ("standard", "sample"):
        n = len(fit[role]["dose_stats"])
        if n < need:
            return (n, need)
    return None


def _make_check(code, severity, message, fn):
    return RuleCheck(code=code, severity=severity, message=message, fn=fn)


# ---------------------------------------------------------------------------
# 默认规则集（版本演进示例）
# ---------------------------------------------------------------------------

_COMMON_CHECKS_V1 = [
    _make_check(
        "FIT_CONVERGED", "invalid", "四参数曲线拟合必须收敛", check_convergence
    ),
    _make_check(
        "PARALLELISM_P",
        "invalid",
        "样品与标准品曲线须平行（共斜率/自由斜率 F 检验 p≥0.05）",
        check_parallelism,
    ),
    _make_check(
        "MIN_DOSE_LEVELS",
        "invalid",
        "标准品与样品均至少 4 个剂量水平",
        check_min_dose_levels,
    ),
    _make_check(
        "MIN_REPLICATES",
        "invalid",
        "每个剂量至少 2 个重复孔",
        check_min_replicates,
    ),
    _make_check(
        "STD_REPLICATE_CV",
        "invalid",
        "标准品各剂量重复 CV 不得超限",
        _make_dose_cv_check("standard"),
    ),
    _make_check(
        "SMP_REPLICATE_CV",
        "invalid",
        "样品各剂量重复 CV 不得超限",
        _make_dose_cv_check("sample"),
    ),
    _make_check(
        "STD_BACKCALC_RECOVERY",
        "warning",
        "标准品反算回收率应落在规定区间",
        _make_recovery_check("standard"),
    ),
    _make_check(
        "SMP_BACKCALC_RECOVERY",
        "warning",
        "样品反算回收率应落在规定区间",
        _make_recovery_check("sample"),
    ),
    _make_check(
        "QUANT_RANGE",
        "invalid",
        "样品 EC50 必须落在标准品实测剂量（有效）范围内",
        check_quantitative_range,
    ),
]

DEFAULT_RULES: list[RuleSet] = [
    RuleSet(
        code="CELL_POTENCY",
        version="1.0",
        effective_from=date(2024, 1, 1),
        description="细胞法效价会审规则 v1.0（初始放行规则）",
        checks=list(_COMMON_CHECKS_V1)
        + [
            _make_check(
                "POTENCY_CI_WIDTH",
                "warning",
                "效价 95% CI 半宽建议 ≤25%",
                check_potency_ci_width,
            )
        ],
        release_low_pct=80.0,
        release_high_pct=125.0,
    ),
    RuleSet(
        code="CELL_POTENCY",
        version="1.1",
        effective_from=date(2026, 1, 1),
        description="细胞法效价会审规则 v1.1（收紧 CI 精密度，CI 超限改判无效）",
        checks=list(_COMMON_CHECKS_V1)
        + [
            _make_check(
                "POTENCY_CI_WIDTH",
                "invalid",
                "效价 95% CI 半宽必须 ≤20%",
                check_potency_ci_width,
            )
        ],
        release_low_pct=80.0,
        release_high_pct=125.0,
    ),
]

# 规则参数上下文（限值集中管理，随规则集版本演进）
DEFAULT_CONTEXT = {
    "parallel_p_min": 0.05,
    "ci_half_width_max_pct": 20.0,
    "max_replicate_cv_pct": 20.0,
    "recovery_pct_range": (80.0, 120.0),
    "min_replicates_per_dose": 2,
    "min_dose_levels": 4,
}

# v1.0 历史参数（CI 限 25%，仅警告由规则严重级别表达；限值保持上下文一致）
_CONTEXT_BY_VERSION = {
    "1.0": {**DEFAULT_CONTEXT, "ci_half_width_max_pct": 25.0},
    "1.1": DEFAULT_CONTEXT,
}


@dataclass
class PlateEvaluation:
    ruleset_code: str
    ruleset_version: str
    valid: bool
    release_passed: Optional[bool]
    potency_percent: Optional[float]
    hits: list = field(default_factory=list)  # list[RuleHit]

    def to_dict(self) -> dict:
        return {
            "ruleset_code": self.ruleset_code,
            "ruleset_version": self.ruleset_version,
            "valid": self.valid,
            "release_passed": self.release_passed,
            "potency_percent": self.potency_percent,
            "hits": [h.to_dict() for h in self.hits],
        }


def context_for(ruleset: RuleSet) -> dict:
    return _CONTEXT_BY_VERSION.get(ruleset.version, DEFAULT_CONTEXT)


def evaluate_plate(
    fit_result: dict,
    ruleset: RuleSet,
    potency_override: Optional[float] = None,
) -> PlateEvaluation:
    """按给定规则集评估一次拟合结果。

    ``fit_result`` 为 ``RelativePotencyResult.to_dict()``；纯函数，可对
    任意历史版本重放。放行判定基于相对效价占标示效价百分比。
    """
    ctx = context_for(ruleset)
    hits: list[RuleHit] = []
    for chk in ruleset.checks:
        try:
            outcome = chk.fn(fit_result, ctx)
        except (KeyError, TypeError, ValueError):
            outcome = (None, None)
        if outcome is not None:
            observed, limit = outcome
            hits.append(
                RuleHit(
                    code=chk.code,
                    severity=chk.severity,
                    message=chk.message,
                    observed=observed,
                    limit=list(limit) if isinstance(limit, tuple) else limit,
                )
            )

    valid = not any(h.severity == "invalid" for h in hits)
    potency = fit_result.get("relative_potency")
    assigned = fit_result.get("assigned_potency")
    pct = None
    release_passed = None
    if potency is not None and assigned:
        pct = potency / assigned * 100.0
        if valid:
            release_passed = ruleset.release_low_pct <= pct <= ruleset.release_high_pct
    return PlateEvaluation(
        ruleset_code=ruleset.code,
        ruleset_version=ruleset.version,
        valid=valid,
        release_passed=release_passed,
        potency_percent=pct,
        hits=hits,
    )


def ruleset_effective_on(rulesets: list[RuleSet], on_date: date, code: str = "CELL_POTENCY"):
    """选出指定日期生效的规则集（该日期 >= effective_from 的最新版本）。"""
    candidates = [
        r for r in rulesets if r.code == code and r.effective_from <= on_date
    ]
    if not candidates:
        raise ValueError(f"日期 {on_date} 没有生效的规则集 {code}")
    return max(candidates, key=lambda r: r.effective_from)
