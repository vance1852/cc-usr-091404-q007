"""版本化规则集：无效板判定。

规则集随实验方法版本保存，分析时快照进分析记录 —— 任何历史分析都可按
"当时生效的规则集"原样复算。每条规则输出一条 RuleHit（命中记录），
全部通过时该测定（板 × 样品系列）判定为有效。
"""
from __future__ import annotations

# 默认规则集（创建方法版本时可覆盖任意字段）
DEFAULT_RULESET: dict = {
    "name": "默认细胞法效价规则集",
    "require_convergence": True,      # R01 两条曲线必须收敛
    "min_span": 0.05,                 # R02 响应窗口 |A-D| 下限
    "min_r_squared": 0.98,            # R03 拟合优度下限
    "max_replicate_rsd_pct": 15.0,    # R04 复孔精密度上限（%）
    "min_valid_levels": 3,            # R05 有效范围内稀释水平数下限
    "parallelism_alpha": 0.05,        # R06 平行性检验显著性水平（p < alpha 判不平行）
    "valid_range_low": 0.1,           # 有效范围窗口下沿（占 A→D 跨度比例）
    "valid_range_high": 0.9,          # 有效范围窗口上沿
}


def merged_ruleset(ruleset: dict | None) -> dict:
    """与默认规则集合并，缺失字段取默认值。"""
    out = dict(DEFAULT_RULESET)
    if ruleset:
        out.update(ruleset)
    return out


def _hit(rule_id: str, description: str, passed: bool, value, threshold, detail: str = "") -> dict:
    return {
        "rule_id": rule_id,
        "description": description,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_ruleset(ruleset: dict | None, ctx: dict) -> list[dict]:
    """按规则集逐条评估。ctx 字段见 services._build_rule_context。

    返回 RuleHit 列表；全部 passed=True 时测定有效。
    """
    rs = merged_ruleset(ruleset)
    hits: list[dict] = []

    std_conv = bool(ctx.get("std_converged"))
    smp_conv = bool(ctx.get("smp_converged"))
    hits.append(_hit(
        "R01_CONVERGENCE", "标准品与样品曲线均须收敛",
        (std_conv and smp_conv) if rs["require_convergence"] else True,
        None, "CONVERGED",
        f"STD={ctx.get('std_status')}, SMP={ctx.get('smp_status')}",
    ))

    span_std = ctx.get("span_std")
    span_smp = ctx.get("span_smp")
    spans = [s for s in (span_std, span_smp) if s is not None]
    min_span_obs = min(spans) if spans else None
    hits.append(_hit(
        "R02_RESPONSE_SPAN", "曲线响应窗口 |A-D| 不得低于下限",
        min_span_obs is not None and min_span_obs >= rs["min_span"],
        min_span_obs, rs["min_span"],
        f"STD span={span_std}, SMP span={span_smp}",
    ))

    r2s = [v for v in (ctx.get("r2_std"), ctx.get("r2_smp")) if v is not None]
    min_r2 = min(r2s) if r2s else None
    hits.append(_hit(
        "R03_R_SQUARED", "拟合优度 R² 不得低于下限",
        min_r2 is not None and min_r2 >= rs["min_r_squared"],
        min_r2, rs["min_r_squared"],
        f"STD R²={ctx.get('r2_std')}, SMP R²={ctx.get('r2_smp')}",
    ))

    rsds = [v for v in (ctx.get("rsd_std"), ctx.get("rsd_smp")) if v is not None]
    max_rsd = max(rsds) if rsds else None
    hits.append(_hit(
        "R04_PRECISION", "复孔 RSD 不得高于上限",
        max_rsd is not None and max_rsd <= rs["max_replicate_rsd_pct"],
        max_rsd, rs["max_replicate_rsd_pct"],
        f"STD maxRSD={ctx.get('rsd_std')}%, SMP maxRSD={ctx.get('rsd_smp')}%",
    ))

    n_std = ctx.get("valid_levels_std")
    n_smp = ctx.get("valid_levels_smp")
    counts = [v for v in (n_std, n_smp) if v is not None]
    min_levels = min(counts) if counts else None
    hits.append(_hit(
        "R05_VALID_RANGE", "有效范围内稀释水平数不得少于下限",
        min_levels is not None and min_levels >= rs["min_valid_levels"],
        min_levels, rs["min_valid_levels"],
        f"STD={n_std} 级, SMP={n_smp} 级",
    ))

    p = ctx.get("parallelism_p")
    hits.append(_hit(
        "R06_PARALLELISM", "平行性检验 p 值不得小于显著性水平",
        p is not None and p >= rs["parallelism_alpha"],
        p, rs["parallelism_alpha"],
        "p 值越小越不平行" if p is not None else "无法计算 p 值",
    ))

    estimable = bool(ctx.get("potency_estimable"))
    hits.append(_hit(
        "R07_ESTIMABLE", "相对效价及其标准误必须可估计",
        estimable, None, "finite",
        "" if estimable else "效价或标准误非有限值",
    ))
    return hits


def all_passed(hits: list[dict]) -> bool:
    return all(h["passed"] for h in hits)
