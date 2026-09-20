"""确定性四参数逻辑（4PL）曲线拟合模块。

模型（以 log10 浓度为自变量）：

    y = D + (A - D) / (1 + 10^(B * (log10(x) - logC)))

参数含义：
    A    —— 低浓度端平台（x → 0 时的响应）
    D    —— 高浓度端平台（x → ∞ 时的响应）
    B    —— Hill 斜率
    logC —— log10(EC50)

数值方法：Levenberg-Marquardt。阻尼因子按固定规则增减，初值完全由数据
确定性导出，全程无随机数、无时间依赖：同一输入在任何时候重算都得到
逐位相同的结果，满足会审复算要求。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

_LN10 = math.log(10.0)
_EXP_CLIP = 150.0  # 防止 10**x 及其平方上溢，固定截断点保证确定性


class FitStatus(str, Enum):
    CONVERGED = "CONVERGED"  # 收敛
    MAX_ITERATIONS = "MAX_ITERATIONS"  # 达到最大迭代次数
    SINGULAR = "SINGULAR"  # 雅可比矩阵奇异，无法求解
    INVALID_DATA = "INVALID_DATA"  # 输入数据不足以拟合


@dataclass
class CurveFit:
    """单条 4PL 曲线的拟合结果。"""

    A: float
    B: float
    logC: float
    D: float
    se_A: float | None
    se_B: float | None
    se_logC: float | None
    se_D: float | None
    status: str
    iterations: int
    rss: float
    r_squared: float
    n_points: int

    @property
    def converged(self) -> bool:
        return self.status == FitStatus.CONVERGED.value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SharedFit:
    """多条曲线共享 A/B/D、各自独立 logC 的约束拟合结果（用于平行性检验）。"""

    A: float
    B: float
    D: float
    logCs: list[float]
    status: str
    iterations: int
    rss: float
    n_points: int

    def to_dict(self) -> dict:
        return asdict(self)


def _model(theta: np.ndarray, lx: np.ndarray) -> np.ndarray:
    A, B, lC, D = theta
    u = np.power(10.0, np.clip(B * (lx - lC), -_EXP_CLIP, _EXP_CLIP))
    return D + (A - D) / (1.0 + u)


def _residuals_jac(theta: np.ndarray, lx: np.ndarray, y: np.ndarray):
    """返回残差向量 r 与雅可比矩阵 J（列序：A, B, logC, D）。"""
    A, B, lC, D = theta
    u = np.power(10.0, np.clip(B * (lx - lC), -_EXP_CLIP, _EXP_CLIP))
    denom = 1.0 + u
    yhat = D + (A - D) / denom
    r = yhat - y
    w = u / (denom * denom)  # u / (1+u)^2
    J = np.empty((lx.size, 4))
    J[:, 0] = 1.0 / denom
    J[:, 1] = -(A - D) * w * (lx - lC) * _LN10
    J[:, 2] = (A - D) * w * B * _LN10
    J[:, 3] = u / denom
    return r, J


def _initial_theta(lx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """由数据确定性导出初值：平台取端部均值方向，logC 取中点线性插值。"""
    order = np.argsort(lx, kind="stable")
    lxs, ys = lx[order], y[order]
    half = max(len(ys) // 2, 1)
    decreasing = bool(ys[:half].mean() >= ys[-half:].mean())
    A0 = float(ys.max() if decreasing else ys.min())
    D0 = float(ys.min() if decreasing else ys.max())
    mid = 0.5 * (A0 + D0)
    lC0 = float(np.median(lxs))
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - mid) * (y1 - mid) <= 0.0 and y0 != y1:
            lC0 = float(lxs[i] + (mid - y0) * (lxs[i + 1] - lxs[i]) / (y1 - y0))
            break
    return np.array([A0, 1.0, lC0, D0])


def _lm_solve(func, theta0: np.ndarray, max_iter: int, tol: float):
    """Levenberg-Marquardt 主循环。阻尼调整规则固定，结果确定。"""
    theta = np.array(theta0, dtype=float)
    r, J = func(theta)
    rss = float(r @ r)
    lam = 1.0e-3
    iterations = 0
    status = FitStatus.MAX_ITERATIONS
    for _ in range(max_iter):
        iterations += 1
        JTJ = J.T @ J
        g = J.T @ r
        scale = np.diag(JTJ).copy()
        scale[scale <= 0.0] = 1.0
        grad_ref = 1.0 + math.sqrt(rss)
        if float(np.max(np.abs(g))) <= 1e-10 * grad_ref:
            status = FitStatus.CONVERGED
            break
        accepted = False
        singular = 0
        while True:
            damped = JTJ + lam * np.diag(scale)
            try:
                delta = np.linalg.solve(damped, -g)
            except np.linalg.LinAlgError:
                singular += 1
                if singular > 15:
                    return theta, rss, iterations, FitStatus.SINGULAR
                lam *= 10.0
                continue
            trial = theta + delta
            r_t, J_t = func(trial)
            rss_t = float(r_t @ r_t)
            if math.isfinite(rss_t) and rss_t < rss:
                prev = rss
                step = float(np.max(np.abs(delta) / (np.abs(theta) + 1e-30)))
                theta, r, J, rss = trial, r_t, J_t, rss_t
                lam = max(lam / 3.0, 1e-12)
                accepted = True
                if (prev - rss) <= tol * (1.0 + prev) and step <= 1e-8:
                    status = FitStatus.CONVERGED
                break
            lam *= 5.0
            if lam > 1e13:
                break
        if status is FitStatus.CONVERGED:
            break
        if not accepted:
            # 阻尼已无法带来下降：梯度足够小视为收敛，否则报告未收敛
            status = (
                FitStatus.CONVERGED
                if float(np.max(np.abs(g))) <= 1e-8 * grad_ref
                else FitStatus.MAX_ITERATIONS
            )
            break
    return theta, rss, iterations, status


def _standard_errors(J: np.ndarray, rss: float, n: int, p: int) -> list[float | None]:
    if n <= p:
        return [None] * p
    s2 = rss / (n - p)
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JTJ)
    out: list[float | None] = []
    for i in range(p):
        v = cov[i, i] * s2
        out.append(math.sqrt(v) if v >= 0.0 and math.isfinite(v) else None)
    return out


def fit_4pl(
    concentrations,
    responses,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> CurveFit:
    """拟合单条 4PL 曲线。concentrations 必须为正，responses 为对应读数。"""
    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(responses, dtype=float)
    n = int(x.size)
    invalid = (
        n < 4
        or y.size != n
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or bool(np.any(x <= 0.0))
    )
    if invalid:
        return CurveFit(
            A=float("nan"), B=float("nan"), logC=float("nan"), D=float("nan"),
            se_A=None, se_B=None, se_logC=None, se_D=None,
            status=FitStatus.INVALID_DATA.value, iterations=0,
            rss=float("nan"), r_squared=float("nan"), n_points=n,
        )
    lx = np.log10(x)

    def func(theta):
        return _residuals_jac(theta, lx, y)

    theta0 = _initial_theta(lx, y)
    theta, rss, iterations, status = _lm_solve(func, theta0, max_iter, tol)
    _, J_final = func(theta)
    se = _standard_errors(J_final, rss, n, 4)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss > 0.0:
        r_squared = 1.0 - rss / tss
    else:
        r_squared = 1.0 if rss == 0.0 else 0.0
    return CurveFit(
        A=float(theta[0]), B=float(theta[1]), logC=float(theta[2]), D=float(theta[3]),
        se_A=se[0], se_B=se[1], se_logC=se[2], se_D=se[3],
        status=status.value, iterations=iterations, rss=float(rss),
        r_squared=float(r_squared), n_points=n,
    )


def fit_shared_asymptotes(
    datasets,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> SharedFit:
    """多条曲线共享 A/B/D、各自独立 logC 的约束拟合（平行性检验用）。

    datasets: [(concentrations, responses), ...]
    """
    blocks = []
    for x, y in datasets:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.size < 2 or ya.size != xa.size or bool(np.any(xa <= 0.0)):
            return SharedFit(
                A=float("nan"), B=float("nan"), D=float("nan"), logCs=[],
                status=FitStatus.INVALID_DATA.value, iterations=0,
                rss=float("nan"), n_points=int(xa.size),
            )
        blocks.append((np.log10(xa), ya))
    k = len(blocks)
    n_total = int(sum(b[0].size for b in blocks))
    # 初值：各曲线独立拟合参数的均值（确定性）
    singles = [fit_4pl(10.0 ** lx, y, max_iter=max_iter, tol=tol) for lx, y in blocks]
    if any(s.status == FitStatus.INVALID_DATA.value for s in singles):
        return SharedFit(
            A=float("nan"), B=float("nan"), D=float("nan"), logCs=[],
            status=FitStatus.INVALID_DATA.value, iterations=0,
            rss=float("nan"), n_points=n_total,
        )
    A0 = float(np.mean([s.A for s in singles]))
    B0 = float(np.mean([s.B for s in singles]))
    D0 = float(np.mean([s.D for s in singles]))
    theta0 = np.array([A0, B0, D0] + [s.logC for s in singles], dtype=float)

    def func(theta: np.ndarray):
        A, B, D = theta[0], theta[1], theta[2]
        lCs = theta[3:]
        r_parts = []
        j_rows = []
        for i, (lx, y) in enumerate(blocks):
            lC = lCs[i]
            u = np.power(10.0, np.clip(B * (lx - lC), -_EXP_CLIP, _EXP_CLIP))
            denom = 1.0 + u
            yhat = D + (A - D) / denom
            r_parts.append(yhat - y)
            w = u / (denom * denom)
            Ji = np.zeros((lx.size, 3 + k))
            Ji[:, 0] = 1.0 / denom
            Ji[:, 1] = -(A - D) * w * (lx - lC) * _LN10
            Ji[:, 2] = u / denom
            Ji[:, 3 + i] = (A - D) * w * B * _LN10
            j_rows.append(Ji)
        return np.concatenate(r_parts), np.vstack(j_rows)

    theta, rss, iterations, status = _lm_solve(func, theta0, max_iter, tol)
    return SharedFit(
        A=float(theta[0]), B=float(theta[1]), D=float(theta[2]),
        logCs=[float(v) for v in theta[3:]],
        status=status.value, iterations=iterations,
        rss=float(rss), n_points=n_total,
    )
