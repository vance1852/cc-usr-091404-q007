"""指标模块测试：精密度、有效范围、平行性、相对效价、组合策略。"""
import math

import pytest

from potency.combine import combine_assay
from potency.fitting import fit_4pl
from potency.metrics import (
    group_by_level,
    parallelism_test,
    relative_potency,
    replicate_rsd,
    valid_levels,
)
from tests.conftest import CONCENTRATIONS, STD_A, STD_B, STD_D, STD_LC, curve_response


def series(logC, B=STD_B, noise=0.0):
    xs, ys = [], []
    for i, c in enumerate(CONCENTRATIONS):
        for rep in range(3):
            y = curve_response(c, STD_A, B, logC, STD_D)
            if noise:
                y *= 1 + noise * math.sin(i * 2.7 + rep * 1.9 + 0.3)
            xs.append(c)
            ys.append(y)
    return xs, ys


class TestReplicateRsd:
    def test_known_value(self):
        groups = {0: [100.0, 102.0, 98.0], 1: [50.0, 50.0, 50.0]}
        out = replicate_rsd(groups)
        mean = 100.0
        sd = math.sqrt(((100 - mean) ** 2 + (102 - mean) ** 2 + (98 - mean) ** 2) / 2)
        assert out["per_level"]["0"] == pytest.approx(sd / 100 * 100)
        assert out["per_level"]["1"] == 0.0
        assert out["max_rsd_pct"] == pytest.approx(out["per_level"]["0"])

    def test_single_replicate_is_zero(self):
        out = replicate_rsd({0: [1.5]})
        assert out["max_rsd_pct"] == 0.0


class TestValidLevels:
    def test_window(self):
        groups = {i: [curve_response(c, STD_A, STD_B, STD_LC, STD_D)] for i, c in enumerate(CONCENTRATIONS)}
        levels = valid_levels(groups, STD_A, STD_D, 0.1, 0.9)
        # 窗口 [0.29, 1.81]：0.1/0.32/1.0/3.2 四级在内
        assert levels == [2, 3, 4, 5]

    def test_flat_curve_no_levels(self):
        groups = {0: [1.0], 1: [1.0], 2: [1.0]}
        # A == D 时窗口收缩为单点，均值恰好在窗口边界上
        assert valid_levels(groups, 1.0, 1.0, 0.1, 0.9) == [0, 1, 2]
        assert valid_levels(groups, 2.0, 1.0, 0.1, 0.9) == []


class TestParallelism:
    def test_parallel_curves_pass(self):
        x1, y1 = series(STD_LC, noise=0.02)
        x2, y2 = series(STD_LC + 0.1, noise=0.02)
        out = parallelism_test(x1, y1, x2, y2)
        assert out["p"] is not None and out["p"] > 0.05
        assert out["df1"] == 3
        assert out["df2"] == len(x1) + len(x2) - 8

    def test_nonparallel_curves_fail(self):
        x1, y1 = series(STD_LC, noise=0.01)
        x2, y2 = series(STD_LC + 0.1, B=2.2, noise=0.01)
        out = parallelism_test(x1, y1, x2, y2)
        assert out["p"] is not None and out["p"] < 0.05


class TestRelativePotency:
    def test_known_ratio(self):
        x1, y1 = series(STD_LC, noise=0.01)
        x2, y2 = series(STD_LC - math.log10(0.8), noise=0.01)
        f1, f2 = fit_4pl(x1, y1), fit_4pl(x2, y2)
        pot = relative_potency(f1, f2)
        assert pot["estimable"]
        assert pot["potency_pct"] == pytest.approx(80.0, abs=1.0)
        lo, hi = pot["ci95_pct"]
        assert lo < 80.0 < hi

    def test_df(self):
        x1, y1 = series(STD_LC)
        x2, y2 = series(STD_LC)
        pot = relative_potency(fit_4pl(x1, y1), fit_4pl(x2, y2))
        assert pot["df"] == len(x1) + len(x2) - 8


class TestCombine:
    def test_mean(self):
        items = [
            {"log_potency": math.log10(0.8), "se": 0.01, "df": 28},
            {"log_potency": math.log10(0.9), "se": 0.01, "df": 28},
        ]
        out = combine_assay(items, "mean")
        assert out["n_determinations"] == 2
        assert out["potency_pct"] == pytest.approx(10 ** ((math.log10(0.8) + math.log10(0.9)) / 2) * 100)
        assert out["ci95_pct"][0] < out["potency_pct"] < out["ci95_pct"][1]

    def test_inverse_variance_weights(self):
        items = [
            {"log_potency": 0.0, "se": 0.01, "df": 10},
            {"log_potency": 0.1, "se": 0.10, "df": 10},
        ]
        out = combine_assay(items, "inverse_variance")
        # 权重 10000:100 → 合并值应非常接近第一个测定
        assert out["log_potency"] == pytest.approx(0.1 * 100 / 10100, abs=1e-12)

    def test_median(self):
        items = [{"log_potency": v, "se": 0.01, "df": 10} for v in (-0.1, 0.0, 0.3)]
        out = combine_assay(items, "median")
        assert out["log_potency"] == 0.0
        assert out["ci95_pct"] is None

    def test_single_uses_own_se(self):
        out = combine_assay([{"log_potency": 0.05, "se": 0.02, "df": 20}], "mean")
        assert out["se_log_potency"] == 0.02
        assert out["df"] == 20

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_assay([], "mean")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            combine_assay([{"log_potency": 0, "se": 1, "df": 1}], "trimmed")
