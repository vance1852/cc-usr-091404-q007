"""数值模块测试：4PL、LM 拟合、相对效价、平行性、精密度、确定性。

全部使用确定性输入（无随机种子依赖）。
"""

import math
import unittest

from potency.fitting import (
    fit_curve,
    fit_relative_potency,
    four_pl,
    f_survival,
    inverse_concentration,
    replicate_stats,
    t_critical,
)
from tests.datafactory import make_wells


def _exact_curve():
    """无噪声精确 4PL 数据。"""
    A, B, C, D = 0.2, 1.1, 0.8, 1.5
    doses = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    x, y = [], []
    for d in doses:
        x.extend([d, d])
        y.extend([four_pl(d, A, B, C, D)] * 2)
    return x, y, (A, B, C, D)


class TestFourPL(unittest.TestCase):
    def test_curve_shape(self):
        self.assertAlmostEqual(four_pl(0.8, 0.2, 1.1, 0.8, 1.5), 0.85, places=10)
        # 低剂量趋近 top D，高剂量趋近 bottom A
        self.assertGreater(four_pl(1e-4, 0.2, 1.1, 0.8, 1.5), 1.49)
        self.assertLess(four_pl(1e3, 0.2, 1.1, 0.8, 1.5), 0.21)

    def test_inverse_roundtrip(self):
        for x in (0.1, 0.5, 0.8, 2.0):
            y = four_pl(x, 0.2, 1.1, 0.8, 1.5)
            self.assertAlmostEqual(inverse_concentration(y, 0.2, 1.1, 0.8, 1.5), x, places=9)

    def test_inverse_outside_asymptotes(self):
        with self.assertRaises(ValueError):
            inverse_concentration(0.2, 0.2, 1.1, 0.8, 1.5)  # 恰在渐近线
        with self.assertRaises(ValueError):
            inverse_concentration(1.6, 0.2, 1.1, 0.8, 1.5)


class TestFitCurve(unittest.TestCase):
    def test_recovers_known_parameters(self):
        x, y, (A, B, C, D) = _exact_curve()
        fit = fit_curve(x, y)
        self.assertTrue(fit.converged)
        for got, want in zip((fit.A, fit.B, fit.C, fit.D), (A, B, C, D)):
            self.assertAlmostEqual(got, want, places=5)
        # 无噪声数据残差接近零
        self.assertLess(fit.sse, 1e-15)

    def test_requires_four_dose_levels(self):
        with self.assertRaises(ValueError):
            fit_curve([1, 1, 2, 2, 3, 3], [0.9] * 6)

    def test_dose_stats_and_quant_range(self):
        x, y, _ = _exact_curve()
        fit = fit_curve(x, y)
        self.assertEqual(len(fit.dose_stats), 7)
        self.assertEqual(fit.qmin, 0.0625)
        self.assertEqual(fit.qmax, 4.0)
        for ds in fit.dose_stats:
            self.assertEqual(ds.n, 2)
            self.assertIsNotNone(ds.recovery_percent)
            self.assertAlmostEqual(ds.recovery_percent, 100.0, places=4)

    def test_replicate_cv(self):
        wells = make_wells(sample_ec50=0.85)
        x = [w["dose"] for w in wells if w["role"] == "standard"]
        y = [w["reading"] for w in wells if w["role"] == "standard"]
        fit = fit_curve(x, y)
        self.assertIsNotNone(fit.pooled_cv_percent)
        self.assertGreaterEqual(fit.max_cv_percent, fit.pooled_cv_percent - 1e-9)
        self.assertGreaterEqual(fit.max_cv_percent, 0.0)


class TestRelativePotency(unittest.TestCase):
    def _xy(self, ec50=0.85, hill=1.1):
        wells = make_wells(sample_ec50=ec50, hill=hill)
        xs = [w["dose"] for w in wells if w["role"] == "standard"]
        ys = [w["reading"] for w in wells if w["role"] == "standard"]
        xm = [w["dose"] for w in wells if w["role"] == "sample"]
        ym = [w["reading"] for w in wells if w["role"] == "sample"]
        return xs, ys, xm, ym

    def test_potency_near_expected_ratio(self):
        # 标准 EC50=0.8，样品 EC50=1.0 → 相对效价 ≈ 80%
        xs, ys, xm, ym = self._xy(ec50=1.0)
        r = fit_relative_potency(xs, ys, xm, ym, 100.0)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.relative_potency, 80.0, delta=1.5)
        # CI 包含点估计
        self.assertLess(r.ci_low, r.relative_potency)
        self.assertGreater(r.ci_high, r.relative_potency)
        # 几何对称性：low*high ≈ potency^2
        self.assertAlmostEqual(
            math.log(r.ci_low) + math.log(r.ci_high),
            2 * math.log(r.relative_potency),
            places=6,
        )

    def test_parallel_data_passes(self):
        xs, ys, xm, ym = self._xy(ec50=0.85)
        r = fit_relative_potency(xs, ys, xm, ym, 100.0)
        self.assertGreater(r.parallel_p, 0.05)
        self.assertTrue(r.parallel_passed)
        self.assertGreaterEqual(r.parallel_f, 0.0)

    def test_nonparallel_data_fails(self):
        xs, ys, xm, ym = self._xy(ec50=0.85)
        # 扭曲样品高剂量端制造斜率差异
        xm2, ym2 = [], []
        for d, v in zip(xm, ym):
            xm2.append(d)
            ym2.append(v + 0.25 * d**2 / (d**2 + 1))
        r = fit_relative_potency(xs, ys, xm2, ym2, 100.0, parallel_p_min=0.05)
        self.assertLess(r.parallel_p, 0.05)
        self.assertFalse(r.parallel_passed)

    def test_invalid_assigned_potency(self):
        xs, ys, xm, ym = self._xy()
        with self.assertRaises(ValueError):
            fit_relative_potency(xs, ys, xm, ym, 0.0)


class TestDeterminism(unittest.TestCase):
    def test_repeat_calls_bit_identical(self):
        wells = make_wells(sample_ec50=0.9)
        xs = [w["dose"] for w in wells if w["role"] == "standard"]
        ys = [w["reading"] for w in wells if w["role"] == "standard"]
        xm = [w["dose"] for w in wells if w["role"] == "sample"]
        ym = [w["reading"] for w in wells if w["role"] == "sample"]
        a = fit_relative_potency(xs, ys, xm, ym, 100.0).to_dict()
        b = fit_relative_potency(xs, ys, xm, ym, 100.0).to_dict()
        self.assertEqual(a, b)

    def test_input_order_independent(self):
        wells = make_wells(sample_ec50=0.9)
        xs = [w["dose"] for w in wells if w["role"] == "standard"]
        ys = [w["reading"] for w in wells if w["role"] == "standard"]
        xm = [w["dose"] for w in wells if w["role"] == "sample"]
        ym = [w["reading"] for w in wells if w["role"] == "sample"]
        ref = fit_relative_potency(xs, ys, xm, ym, 100.0).to_dict()
        order = list(range(len(xs)))[::-1]
        rev = fit_relative_potency(
            [xs[i] for i in order], [ys[i] for i in order],
            [xm[i] for i in order], [ym[i] for i in order], 100.0,
        ).to_dict()
        self.assertEqual(ref, rev)


class TestStats(unittest.TestCase):
    def test_f_distribution_table_values(self):
        # F 临界值上尾概率（与标准 F 表对照）
        self.assertAlmostEqual(f_survival(2.901, 5, 15), 0.05, delta=1e-3)
        self.assertAlmostEqual(f_survival(4.556, 5, 15), 0.01, delta=1e-3)
        self.assertAlmostEqual(f_survival(2.35, 10, 19), 0.0523, delta=1e-3)
        self.assertEqual(f_survival(0.0, 2, 10), 1.0)
        self.assertLess(f_survival(100.0, 2, 10), 1e-6)

    def test_t_critical(self):
        self.assertAlmostEqual(t_critical(10), 2.228, delta=0.001)
        self.assertAlmostEqual(t_critical(1000), 1.962, delta=0.002)

    def test_replicate_stats(self):
        x, y, _ = _exact_curve()
        stats = replicate_stats(x, y)
        self.assertEqual(len(stats), 7)
        self.assertAlmostEqual(stats[3]["recovery_percent"], 100.0, places=4)


if __name__ == "__main__":
    unittest.main()
