"""4PL 拟合模块测试：精度、收敛状态与确定性。"""
import math

import pytest

from potency.fitting import FitStatus, fit_4pl, fit_shared_asymptotes
from tests.conftest import CONCENTRATIONS, STD_A, STD_B, STD_D, STD_LC, curve_response


def clean_data(A=STD_A, B=STD_B, logC=STD_LC, D=STD_D):
    xs, ys = [], []
    for c in CONCENTRATIONS:
        for _ in range(3):
            xs.append(c)
            ys.append(curve_response(c, A, B, logC, D))
    return xs, ys


class TestCleanData:
    def test_recovers_parameters(self):
        x, y = clean_data()
        fit = fit_4pl(x, y)
        assert fit.status == FitStatus.CONVERGED.value
        assert fit.converged
        assert fit.A == pytest.approx(STD_A, abs=1e-6)
        assert fit.B == pytest.approx(STD_B, abs=1e-6)
        assert fit.logC == pytest.approx(STD_LC, abs=1e-6)
        assert fit.D == pytest.approx(STD_D, abs=1e-6)
        assert fit.rss == pytest.approx(0.0, abs=1e-20)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_standard_errors_tiny_on_clean_data(self):
        x, y = clean_data()
        fit = fit_4pl(x, y)
        assert fit.se_logC is not None and fit.se_logC < 1e-4


class TestNoisyData:
    def test_converges_and_recovers_approximately(self):
        xs, ys = [], []
        for i, c in enumerate(CONCENTRATIONS):
            for rep in range(3):
                xs.append(c)
                ys.append(curve_response(c, STD_A, STD_B, STD_LC, STD_D)
                          * (1 + 0.02 * math.sin(i * 2.7 + rep * 1.9)))
        fit = fit_4pl(xs, ys)
        assert fit.converged
        assert fit.A == pytest.approx(STD_A, abs=0.05)
        assert fit.logC == pytest.approx(STD_LC, abs=0.02)
        assert fit.r_squared > 0.99


class TestDeterminism:
    def test_bit_identical_on_recompute(self):
        x, y = clean_data(B=1.3, logC=-0.2)
        f1 = fit_4pl(x, y)
        f2 = fit_4pl(x, y)
        assert f1.to_dict() == f2.to_dict()

    def test_increasing_curve(self):
        # 上升型曲线（A < D）也应收敛
        xs, ys = clean_data(A=0.1, D=2.0)
        fit = fit_4pl(xs, ys)
        assert fit.converged
        assert fit.A == pytest.approx(0.1, abs=1e-5)
        assert fit.D == pytest.approx(2.0, abs=1e-5)


class TestInvalidData:
    def test_too_few_points(self):
        fit = fit_4pl([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert fit.status == FitStatus.INVALID_DATA.value

    def test_nonpositive_concentration(self):
        fit = fit_4pl([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        assert fit.status == FitStatus.INVALID_DATA.value

    def test_nan_input(self):
        fit = fit_4pl([1.0, 2.0, 3.0, 4.0], [1.0, float("nan"), 3.0, 4.0])
        assert fit.status == FitStatus.INVALID_DATA.value


class TestSharedFit:
    def test_recovers_shared_parameters(self):
        x1, y1 = clean_data(logC=STD_LC)
        x2, y2 = clean_data(logC=STD_LC + 0.1)
        shared = fit_shared_asymptotes([(x1, y1), (x2, y2)])
        assert shared.status == FitStatus.CONVERGED.value
        assert shared.A == pytest.approx(STD_A, abs=1e-5)
        assert shared.B == pytest.approx(STD_B, abs=1e-5)
        assert shared.D == pytest.approx(STD_D, abs=1e-5)
        assert shared.logCs[0] == pytest.approx(STD_LC, abs=1e-5)
        assert shared.logCs[1] == pytest.approx(STD_LC + 0.1, abs=1e-5)

    def test_constrained_rss_not_smaller(self):
        # 约束模型的 RSS 不可能小于独立拟合之和
        from potency.fitting import fit_4pl as f4
        x1, y1 = clean_data(logC=STD_LC)
        x2, y2 = clean_data(logC=STD_LC + 0.1, B=1.4)
        shared = fit_shared_asymptotes([(x1, y1), (x2, y2)])
        rss_u = f4(x1, y1).rss + f4(x2, y2).rss
        assert shared.rss >= rss_u - 1e-12
