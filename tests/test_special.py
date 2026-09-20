"""确定性特殊函数测试：与 scipy 交叉验证。"""
import math

import pytest
from scipy.special import betainc as scipy_betainc
from scipy.stats import f as scipy_f
from scipy.stats import t as scipy_t

from potency.special import betai, f_sf, t_ppf


class TestBetai:
    def test_boundary(self):
        assert betai(1.0, 1.0, 0.0) == 0.0
        assert betai(1.0, 1.0, 1.0) == 1.0

    def test_uniform_case(self):
        # I_x(1,1) = x
        for x in (0.1, 0.25, 0.5, 0.9):
            assert betai(1.0, 1.0, x) == pytest.approx(x, abs=1e-12)

    @pytest.mark.parametrize("a,b,x", [
        (0.5, 0.5, 0.3), (2.0, 3.0, 0.4), (3.0, 1.5, 0.7),
        (14.0, 14.0, 0.5), (1.0, 8.0, 0.15), (9.0, 2.0, 0.82),
    ])
    def test_against_scipy(self, a, b, x):
        assert betai(a, b, x) == pytest.approx(scipy_betainc(a, b, x), rel=1e-12, abs=1e-14)


class TestFDistribution:
    @pytest.mark.parametrize("fval,d1,d2", [
        (0.5, 3, 28), (1.0, 3, 28), (2.93, 3, 28), (5.0, 1, 10), (0.1, 4, 40),
    ])
    def test_against_scipy(self, fval, d1, d2):
        assert f_sf(fval, d1, d2) == pytest.approx(scipy_f.sf(fval, d1, d2), rel=1e-10)

    def test_zero(self):
        assert f_sf(0.0, 3, 28) == 1.0


class TestTDistribution:
    @pytest.mark.parametrize("p,df", [(0.975, 28), (0.975, 3), (0.995, 10), (0.5, 5), (0.025, 20)])
    def test_against_scipy(self, p, df):
        assert t_ppf(p, df) == pytest.approx(scipy_t.ppf(p, df), rel=1e-9)

    def test_symmetry(self):
        assert t_ppf(0.025, 12) == pytest.approx(-t_ppf(0.975, 12), rel=1e-12)

    def test_invalid(self):
        with pytest.raises(ValueError):
            t_ppf(0.0, 5)
        with pytest.raises(ValueError):
            t_ppf(0.5, 0.5)
