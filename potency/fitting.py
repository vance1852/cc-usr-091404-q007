"""确定性数值计算模块。

纯 Python 标准库实现，无第三方依赖：

* 四参数逻辑曲线 (4PL)：``y = A + (D - A) / (1 + (x/C)^B)``
* Levenberg–Marquardt 非线性最小二乘（固定迭代策略，浮点结果可逐位复现）
* 相对效价：标准品/样品共用斜率的 EC50 比值法
* 平行性：约束（共斜率）与自由模型的 F 检验
* 精密度：孔内重复 SD / CV%、合并 CV
* 有效范围与反算回收率
* 效价置信区间（delta 法，对数正态近似）

所有函数对输入顺序不敏感（内部按剂量排序），不使用线程、随机数或
哈希迭代序，保证同一输入永远得到同一输出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean 需要至少一个值")
    return sum(values) / len(values)


def sample_sd(values: Sequence[float]) -> Optional[float]:
    """样本标准差；n<2 时返回 None。"""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def geom_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geom_mean 需要至少一个值")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def geom_cv_percent(values: Sequence[float]) -> Optional[float]:
    """几何变异系数：sqrt(exp(s^2)-1)*100，s 为对数 SD。"""
    n = len(values)
    if n < 2:
        return None
    logs = [math.log(v) for v in values]
    m = sum(logs) / n
    s2 = sum((v - m) ** 2 for v in logs) / (n - 1)
    return math.sqrt(math.exp(s2) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# 4PL 模型
# ---------------------------------------------------------------------------


def four_pl(x: float, A: float, B: float, C: float, D: float) -> float:
    """y = A + (D-A)/(1+(x/C)^B)。

    A 为高浓度渐近下限（bottom），D 为低浓度渐近上限（top），
    C 为半效浓度 EC50，B 为 Hill 斜率（下降曲线 B>0）。
    """
    if x <= 0 or C <= 0:
        raise ValueError("4PL 剂量与 EC50 必须为正数")
    return A + (D - A) / (1.0 + (x / C) ** B)


def inverse_concentration(y: float, A: float, B: float, C: float, D: float) -> float:
    """由响应反算浓度 x；响应必须严格落在 (A, D) 之间。"""
    if not (min(A, D) < y < max(A, D)):
        raise ValueError(f"响应 {y!r} 落在渐近线 [{A!r}, {D!r}] 之外，无法反算")
    ratio = (D - A) / (y - A) - 1.0
    if ratio <= 0:
        raise ValueError("反算比值非正")
    return C * ratio ** (1.0 / B)


@dataclass
class DoseStats:
    dose: float
    n: int
    mean: float
    sd: Optional[float]
    cv_percent: Optional[float]
    fitted: float
    backcalc: Optional[float]
    recovery_percent: Optional[float]


@dataclass
class CurveFit:
    role: str
    converged: bool
    iterations: int
    A: float
    B: float
    C: float
    D: float
    se: dict = field(default_factory=dict)  # 参数名 -> 标准误
    sse: float = float("nan")
    df: int = 0
    residual_sd: Optional[float] = None
    warnings: list = field(default_factory=list)
    dose_stats: list = field(default_factory=list)  # list[DoseStats]
    qmin: Optional[float] = None
    qmax: Optional[float] = None
    pooled_cv_percent: Optional[float] = None
    max_cv_percent: Optional[float] = None
    min_recovery_percent: Optional[float] = None
    max_recovery_percent: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "role": self.role,
            "converged": self.converged,
            "iterations": self.iterations,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "ec50": self.C,
            "hill_slope": self.B,
            "se": self.se,
            "sse": self.sse,
            "df": self.df,
            "residual_sd": self.residual_sd,
            "warnings": list(self.warnings),
            "quantitative_range": [self.qmin, self.qmax],
            "pooled_cv_percent": self.pooled_cv_percent,
            "max_cv_percent": self.max_cv_percent,
            "min_recovery_percent": self.min_recovery_percent,
            "max_recovery_percent": self.max_recovery_percent,
            "dose_stats": [ds.__dict__ for ds in self.dose_stats],
        }
        return d


@dataclass
class RelativePotencyResult:
    converged: bool
    iterations: int
    common_slope: float
    assigned_potency: float
    relative_potency: float
    ratio_ec50: float
    se_log_ratio: float
    ci_low: float
    ci_high: float
    parallel_f: float
    parallel_df1: int
    parallel_df2: int
    parallel_p: float
    parallel_passed: bool
    standard: CurveFit
    sample: CurveFit
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "common_slope": self.common_slope,
            "assigned_potency": self.assigned_potency,
            "relative_potency": self.relative_potency,
            "ratio_ec50": self.ratio_ec50,
            "se_log_ratio": self.se_log_ratio,
            "ci95": [self.ci_low, self.ci_high],
            "ci_half_width_percent": (self.ci_high - self.ci_low)
            / (2.0 * self.relative_potency)
            * 100.0,
            "parallelism": {
                "method": "F-test (common vs free slope)",
                "f_statistic": self.parallel_f,
                "df1": self.parallel_df1,
                "df2": self.parallel_df2,
                "p_value": self.parallel_p,
                "passed": self.parallel_passed,
            },
            "standard": self.standard.to_dict(),
            "sample": self.sample.to_dict(),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Levenberg–Marquardt 引擎
# ---------------------------------------------------------------------------

# 每条曲线内部参数顺序：(A, logC, logSpan)；共斜率时 logB 为全局参数。
_TINY = 1e-12
_MAX_ITER = 100
_FTOL = 1e-12


def _initial_curve(x_groups: list[tuple[float, list[float]]]) -> tuple[float, float, float]:
    doses = [g[0] for g in x_groups]
    means = [mean(g[1]) for g in x_groups]
    top = means[0]  # 已按剂量升序：最低剂量响应最高
    bottom = means[-1]
    if bottom >= top:
        # 非单调数据也给一个非零跨度初值
        span = max(abs(top) * 0.1, 1e-6)
        A0 = bottom
    else:
        span = top - bottom
        A0 = bottom
    log_mid = (math.log(doses[0]) + math.log(doses[-1])) / 2.0
    return A0, log_mid, math.log(max(span, 1e-9))


def _unpack(theta: Sequence[float], n_curves: int, common_slope: bool):
    curves = []
    for i in range(n_curves):
        A = theta[3 * i]
        logC = theta[3 * i + 1]
        logSpan = theta[3 * i + 2]
        if common_slope:
            logB = theta[3 * n_curves]
        else:
            logB = theta[3 * n_curves + i]
        curves.append((A, math.exp(logB), math.exp(logC), A + math.exp(logSpan)))
    return curves


def _curve_residuals(x, y, params):
    A, B, C, D = params
    return [four_pl(xi, A, B, C, D) - yi for xi, yi in zip(x, y)]


def _all_residuals(theta, curves_data, common_slope):
    params = _unpack(theta, len(curves_data), common_slope)
    res = []
    for (x, y), p in zip(curves_data, params):
        res.extend(_curve_residuals(x, y, p))
    return res


def _jacobian(theta, curves_data, common_slope):
    """前向差分雅可比（确定性步长）。"""
    npar = len(theta)
    r0 = _all_residuals(theta, curves_data, common_slope)
    n = len(r0)
    J = [[0.0] * npar for _ in range(n)]
    for j in range(npar):
        h = 1e-7 * max(1.0, abs(theta[j]))
        t2 = list(theta)
        t2[j] += h
        r1 = _all_residuals(t2, curves_data, common_slope)
        for i in range(n):
            J[i][j] = (r1[i] - r0[i]) / h
    return J, r0


def _solve_linear(A, b):
    """高斯消元（带部分主元）解 A x = b；奇异返回 None。"""
    n = len(b)
    M = [row[:] for row in A]
    y = b[:]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            y[col], y[piv] = y[piv], y[col]
        pivval = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / pivval
            if factor == 0.0:
                continue
            M[r][col] = 0.0
            for c in range(col + 1, n):
                M[r][c] -= factor * M[col][c]
            y[r] -= factor * y[col]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        if abs(M[i][i]) < 1e-14:
            return None
        x[i] = s / M[i][i]
    return x


def _inverse_matrix(M):
    """正定矩阵求逆（逐列高斯消元）；奇异返回 None。"""
    n = len(M)
    cols = []
    for k in range(n):
        e = [0.0] * n
        e[k] = 1.0
        sol = _solve_linear(M, e)
        if sol is None:
            return None
        cols.append(sol)
    return [[cols[c][r] for c in range(n)] for r in range(n)]


def _levenberg(curves_data, common_slope):
    """返回 dict：theta, params(每曲线 A,B,C,D), cov, sse, df, converged,
    iterations, sse_history。"""
    n_curves = len(curves_data)
    # 规范化点顺序（按剂量、响应排序，保持 x-y 配对），保证残差求和顺序
    # 确定，结果与调用方给出的孔序无关。
    ordered = []
    for cx, cy in curves_data:
        pairs = sorted(zip(cx, cy), key=lambda p: (p[0], p[1]))
        ordered.append(([p[0] for p in pairs], [p[1] for p in pairs]))
    curves_data = ordered
    groups = [_group_points(x, y) for x, y in curves_data]
    theta0 = []
    for g in groups:
        A0, logC0, logSpan0 = _initial_curve(g)
        theta0.extend([A0, logC0, logSpan0])
    theta0.append(0.0 if common_slope else 0.0)  # 占位
    if not common_slope:
        theta0.extend([0.0] * (n_curves - 1))
    # logB 全部置 0（B=1）
    offset = 3 * n_curves
    for i in range(n_curves):
        theta0[offset + (0 if common_slope else i)] = 0.0

    nres = sum(len(x) for x, _ in curves_data)
    npar = len(theta0)

    theta = theta0
    residuals = _all_residuals(theta, curves_data, common_slope)
    sse = sum(r * r for r in residuals)
    lam = 1e-3
    converged = False
    history = [sse]
    iterations = 0

    for it in range(1, _MAX_ITER + 1):
        iterations = it
        J, r0 = _jacobian(theta, curves_data, common_slope)
        # JtJ 与 Jtr
        JtJ = [[0.0] * npar for _ in range(npar)]
        Jtr = [0.0] * npar
        for i in range(nres):
            ri = r0[i]
            Ji = J[i]
            for a in range(npar):
                Jra = Ji[a]
                Jtr[a] -= Jra * ri
                for b in range(a, npar):
                    JtJ[a][b] += Jra * Ji[b]
        for a in range(npar):
            for b in range(a):
                JtJ[a][b] = JtJ[b][a]

        accepted = False
        step = None
        for _trial in range(60):
            A = [row[:] for row in JtJ]
            for a in range(npar):
                A[a][a] += lam * (JtJ[a][a] if JtJ[a][a] > 0 else 1.0)
            step = _solve_linear(A, Jtr)
            if step is None:
                lam *= 10.0
                continue
            t_new = [theta[i] + step[i] for i in range(npar)]
            try:
                r_new = _all_residuals(t_new, curves_data, common_slope)
            except (ValueError, OverflowError):
                lam *= 10.0
                continue
            sse_new = sum(r * r for r in r_new)
            if math.isnan(sse_new) or math.isinf(sse_new):
                lam *= 10.0
                continue
            if sse_new < sse:
                rel = (sse - sse_new) / max(sse, _TINY)
                theta = t_new
                sse = sse_new
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                history.append(sse)
                if rel < _FTOL:
                    converged = True
                break
            lam *= 10.0

        if not accepted:
            # 无法继续下降：接受当前点
            converged = True
            break
        if converged:
            break

    # 最终协方差
    J, r0 = _jacobian(theta, curves_data, common_slope)
    JtJ = [[0.0] * npar for _ in range(npar)]
    for i in range(nres):
        Ji = J[i]
        for a in range(npar):
            for b in range(a, npar):
                JtJ[a][b] += Ji[a] * Ji[b]
    for a in range(npar):
        for b in range(a):
            JtJ[a][b] = JtJ[b][a]
    cov_theta = _inverse_matrix(JtJ)
    df = nres - npar
    if cov_theta is not None and df > 0:
        sigma2 = sse / df
        cov_theta = [[v * sigma2 for v in row] for row in cov_theta]

    params = _unpack(theta, n_curves, common_slope)
    return {
        "theta": theta,
        "params": params,
        "cov_theta": cov_theta,
        "sse": sse,
        "df": df,
        "nres": nres,
        "npar": npar,
        "converged": converged,
        "iterations": iterations,
        "sse_history": history,
    }


def _group_points(x: Sequence[float], y: Sequence[float]) -> list[tuple[float, list[float]]]:
    if len(x) != len(y):
        raise ValueError("x/y 长度不一致")
    groups: dict[float, list[float]] = {}
    for xi, yi in zip(x, y):
        if xi <= 0:
            raise ValueError(f"剂量必须为正数，收到 {xi!r}")
        groups.setdefault(xi, []).append(yi)
    # 剂量升序、组内响应升序：求和顺序确定，结果与孔序无关
    return sorted((dose, sorted(vals)) for dose, vals in groups.items())


def _param_se(theta, cov, index_map, n_curves):
    """通过中心差分把内部参数协方差映射到 (A,B,C,D) 的标准误。"""
    if cov is None:
        return {k: None for k in ("A", "B", "C", "D")}

    def physical(t):
        return _unpack(t, n_curves, index_map["common"])

    base = physical(theta)
    se = []
    for i in range(n_curves):
        row = {}
        for k, name in enumerate(("A", "B", "C", "D")):
            grad = []
            for j in range(len(theta)):
                h = 1e-6 * max(1.0, abs(theta[j]))
                t1 = list(theta)
                t1[j] += h
                t2 = list(theta)
                t2[j] -= h
                try:
                    p1 = physical(t1)[i][k]
                    p2 = physical(t2)[i][k]
                    grad.append((p1 - p2) / (2 * h))
                except (ValueError, OverflowError):
                    grad.append(0.0)
            var = 0.0
            for a in range(len(theta)):
                for b in range(len(theta)):
                    var += grad[a] * grad[b] * cov[a][b]
            row[name] = math.sqrt(var) if var > 0 else None
        se.append(row)
    return se


def _build_curve_fit(role, x, y, fit, curve_index, n_curves, common):
    A, B, C, D = fit["params"][curve_index]
    warnings = []
    im = {"common": common}
    se_rows = _param_se(fit["theta"], fit["cov_theta"], im, n_curves)
    se = se_rows[curve_index]

    groups = _group_points(x, y)
    dose_stats = []
    cvs = []
    for dose, vals in groups:
        m = mean(vals)
        sd = sample_sd(vals)
        cv = (sd / m * 100.0) if (sd is not None and m != 0) else None
        fitted = four_pl(dose, A, B, C, D)
        try:
            bc = inverse_concentration(m, A, B, C, D)
            rec = bc / dose * 100.0
        except ValueError:
            bc, rec = None, None
        if cv is not None:
            cvs.append(cv)
        dose_stats.append(
            DoseStats(
                dose=dose,
                n=len(vals),
                mean=m,
                sd=sd,
                cv_percent=cv,
                fitted=fitted,
                backcalc=bc,
                recovery_percent=rec,
            )
        )
    recs = [ds.recovery_percent for ds in dose_stats if ds.recovery_percent is not None]
    # 合并 CV：基于各组方差加权（自由度 n_i-1）
    pooled = None
    if any(_has_replicate(ds) for ds in dose_stats):
        num = 0.0
        den = 0
        for ds in dose_stats:
            if ds.sd is not None and ds.mean != 0:
                num += (ds.n - 1) * (ds.sd / ds.mean) ** 2
                den += ds.n - 1
        if den > 0:
            pooled = math.sqrt(num / den) * 100.0
    if not fit["converged"]:
        warnings.append("LM_NOT_CONVERGED")
    if fit["df"] <= 0:
        warnings.append("INSUFFICIENT_DEGREES_OF_FREEDOM")

    return CurveFit(
        role=role,
        converged=fit["converged"],
        iterations=fit["iterations"],
        A=A,
        B=B,
        C=C,
        D=D,
        se=se,
        sse=fit["sse"],
        df=max(fit["df"], 0),
        residual_sd=math.sqrt(fit["sse"] / fit["df"]) if fit["df"] > 0 else None,
        warnings=warnings,
        dose_stats=dose_stats,
        qmin=groups[0][0] if groups else None,
        qmax=groups[-1][0] if groups else None,
        pooled_cv_percent=pooled,
        max_cv_percent=max(cvs) if cvs else None,
        min_recovery_percent=min(recs) if recs else None,
        max_recovery_percent=max(recs) if recs else None,
    )


def _has_replicate(ds: DoseStats) -> bool:
    return ds.n >= 2


def _canonical_xy(x: Sequence[float], y: Sequence[float]):
    """按 (剂量, 响应) 排序并保持配对，消除输入顺序对浮点求和的影响。"""
    pairs = sorted(zip(x, y), key=lambda p: (p[0], p[1]))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def fit_curve(x: Sequence[float], y: Sequence[float], role: str = "standard") -> CurveFit:
    """拟单单条 4PL 曲线（至少 4 个不同剂量点）。"""
    x, y = _canonical_xy(x, y)
    groups = _group_points(x, y)
    if len(groups) < 4:
        raise ValueError(f"4PL 至少需要 4 个不同剂量，当前 {len(groups)} 个")
    fit = _levenberg([(x, y)], common_slope=False)
    return _build_curve_fit(role, x, y, fit, 0, 1, False)


fit_four_pl = fit_curve


# ---------------------------------------------------------------------------
# F 分布（正则化不完全 Beta）
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT = 200
    EPS = 3e-14
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_survival(f: float, df1: int, df2: int) -> float:
    """P(F(df1,df2) > f)。"""
    if f <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _betai(df2 / 2.0, df1 / 2.0, x)


def _t_survival(t: float, nu: int) -> float:
    """双侧 P(|T| > t)。"""
    x = nu / (nu + t * t)
    return _betai(nu / 2.0, 0.5, x)


def t_critical(df: int, alpha: float = 0.05) -> float:
    """双侧 t 分位数（二分法求逆，确定性）。"""
    if df < 1:
        return 1.959963984540054
    target = alpha
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_survival(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# 相对效价（共斜率 EC50 比值法）+ 平行性
# ---------------------------------------------------------------------------


def fit_relative_potency(
    x_std: Sequence[float],
    y_std: Sequence[float],
    x_smp: Sequence[float],
    y_smp: Sequence[float],
    assigned_potency: float,
    alpha: float = 0.05,
    parallel_p_min: float = 0.05,
) -> RelativePotencyResult:
    """计算相对效价 Pr = (EC50_std / EC50_smp) × 标准品标示效价。

    共斜率联合拟合给出点估计与 delta 法对数区间；自由斜率模型用于
    F 检验平行性。
    """
    if assigned_potency <= 0:
        raise ValueError("标示效价必须为正数")
    x_std, y_std = _canonical_xy(x_std, y_std)
    x_smp, y_smp = _canonical_xy(x_smp, y_smp)
    if len(_group_points(x_std, y_std)) < 4 or len(_group_points(x_smp, y_smp)) < 4:
        raise ValueError("标准品与样品均至少需要 4 个不同剂量点")

    curves = [(x_std, y_std), (x_smp, y_smp)]
    common = _levenberg(curves, common_slope=True)
    free = _levenberg(curves, common_slope=False)

    (A0, B0, C0, D0), (A1, B1, C1, D1) = common["params"]
    ratio = C0 / C1
    potency = ratio * assigned_potency

    # log ratio = logC_std - logC_smp，内部参数 logC 索引 = 3*i+1
    cov = common["cov_theta"]
    se_log = float("nan")
    if cov is not None:
        i_std, i_smp = 1, 4
        var = cov[i_std][i_std] + cov[i_smp][i_smp] - 2 * cov[i_std][i_smp]
        if var > 0:
            se_log = math.sqrt(var)
    z = t_critical(max(common["df"], 1), alpha) if common["df"] >= 1 else \
        (1.959963984540054 if abs(alpha - 0.05) < 1e-12
         else _normal_quantile(1 - alpha / 2))
    if math.isnan(se_log):
        ci_low = ci_high = float("nan")
    else:
        ci_low = math.exp(math.log(ratio) - z * se_log) * assigned_potency
        ci_high = math.exp(math.log(ratio) + z * se_log) * assigned_potency

    # 平行性 F 检验：约束模型（共斜率）SSE 必 ≥ 自由模型 SSE
    df1 = free["npar"] - common["npar"]  # = 曲线数 - 1 = 1
    df2 = free["df"]
    sse_diff = common["sse"] - free["sse"]
    if df2 > 0 and free["sse"] > 0 and sse_diff >= 0:
        f_stat = (sse_diff / df1) / (free["sse"] / df2)
        p = f_survival(f_stat, df1, df2)
    else:
        f_stat, p = 0.0, 1.0
    parallel_passed = p >= parallel_p_min

    warnings = []
    if not common["converged"] or not free["converged"]:
        warnings.append("LM_NOT_CONVERGED")
    if not (C1 >= min(x_std) and C1 <= max(x_std)):
        warnings.append("SAMPLE_EC50_OUTSIDE_STD_RANGE")

    std_fit = _build_curve_fit("standard", x_std, y_std, common, 0, 2, True)
    smp_fit = _build_curve_fit("sample", x_smp, y_smp, common, 1, 2, True)
    # 自由斜率也记录到警告/供比较
    std_fit.se["B_free"] = None
    smp_fit.se["B_free"] = None
    std_fit.warnings.append(f"free_slope={free['params'][0][1]:.6g}")
    smp_fit.warnings.append(f"free_slope={free['params'][1][1]:.6g}")

    return RelativePotencyResult(
        converged=common["converged"] and free["converged"],
        iterations=common["iterations"] + free["iterations"],
        common_slope=B0,
        assigned_potency=assigned_potency,
        relative_potency=potency,
        ratio_ec50=ratio,
        se_log_ratio=se_log,
        ci_low=ci_low,
        ci_high=ci_high,
        parallel_f=f_stat,
        parallel_df1=df1,
        parallel_df2=max(df2, 0),
        parallel_p=p,
        parallel_passed=parallel_passed,
        standard=std_fit,
        sample=smp_fit,
        warnings=warnings,
    )


def _normal_quantile(p: float) -> float:
    """标准正态分位数（Acklam 近似）。"""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def replicate_stats(x: Sequence[float], y: Sequence[float]) -> list[dict]:
    """供外部调用的剂量级重复统计（含拟合值/反算/回收率，若可拟合）。"""
    groups = _group_points(x, y)
    out = []
    fit = None
    try:
        if len(groups) >= 4:
            fit = fit_curve(x, y)
    except ValueError:
        fit = None
    for dose, vals in groups:
        m = mean(vals)
        sd = sample_sd(vals)
        entry = {
            "dose": dose,
            "n": len(vals),
            "mean": m,
            "sd": sd,
            "cv_percent": (sd / m * 100.0) if (sd is not None and m != 0) else None,
        }
        if fit is not None:
            entry["fitted"] = four_pl(dose, fit.A, fit.B, fit.C, fit.D)
            try:
                entry["backcalc"] = inverse_concentration(m, fit.A, fit.B, fit.C, fit.D)
                entry["recovery_percent"] = entry["backcalc"] / dose * 100.0
            except ValueError:
                entry["backcalc"] = None
                entry["recovery_percent"] = None
        out.append(entry)
    return out
