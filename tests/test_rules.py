"""规则集与放行判定测试。"""

import unittest
from datetime import date

from potency.fitting import fit_relative_potency
from potency.rules import (
    DEFAULT_CONTEXT,
    DEFAULT_RULES,
    evaluate_plate,
    ruleset_effective_on,
)
from tests.datafactory import make_wells

V10 = next(r for r in DEFAULT_RULES if r.version == "1.0")
V11 = next(r for r in DEFAULT_RULES if r.version == "1.1")


def fit_dict(ec50=0.85, distort=False, assigned=100.0):
    wells = make_wells(sample_ec50=ec50)
    xs = [w["dose"] for w in wells if w["role"] == "standard"]
    ys = [w["reading"] for w in wells if w["role"] == "standard"]
    xm = [w["dose"] for w in wells if w["role"] == "sample"]
    ym = [w["reading"] for w in wells if w["role"] == "sample"]
    if distort:
        ym = [v + 0.25 * d**2 / (d**2 + 1) for d, v in zip(xm, ym)]
    return fit_relative_potency(xs, ys, xm, ym, assigned).to_dict()


class TestRuleSets(unittest.TestCase):
    def test_effective_date_selection(self):
        self.assertEqual(
            ruleset_effective_on(DEFAULT_RULES, date(2025, 6, 1)).version, "1.0"
        )
        self.assertEqual(
            ruleset_effective_on(DEFAULT_RULES, date(2026, 1, 1)).version, "1.1"
        )
        self.assertEqual(
            ruleset_effective_on(DEFAULT_RULES, date(2026, 9, 20)).version, "1.1"
        )
        with self.assertRaises(ValueError):
            ruleset_effective_on(DEFAULT_RULES, date(2023, 1, 1))

    def test_good_plate_valid_and_released(self):
        ev = evaluate_plate(fit_dict(0.85), V11)
        self.assertTrue(ev.valid)
        self.assertTrue(ev.release_passed)
        self.assertNotIn(
            "PARALLELISM_P", {h.code for h in ev.hits if h.severity == "invalid"}
        )

    def test_nonparallel_is_invalid_even_if_potency_in_range(self):
        # 扭曲后效价可能落区间内，但非平行必须判废
        ev = evaluate_plate(fit_dict(0.85, distort=True), V11)
        self.assertFalse(ev.valid)
        codes = {h.code for h in ev.hits}
        self.assertIn("PARALLELISM_P", codes)
        # 无效板不产生放行通过结论
        self.assertFalse(ev.release_passed)

    def test_potency_below_release_range(self):
        ev = evaluate_plate(fit_dict(1.05), V11)  # ~77%
        self.assertTrue(ev.valid, "板本身有效，但低于放行限")
        self.assertFalse(ev.release_passed)
        self.assertLess(ev.potency_percent, 80.0)

    def test_all_hits_recorded_not_just_first(self):
        ev = evaluate_plate(fit_dict(0.85, distort=True), V11)
        # 命中是完整列表（可能多条），不做布尔掩盖
        self.assertGreaterEqual(len(ev.hits), 1)
        for h in ev.hits:
            self.assertIn(h.severity, ("invalid", "warning"))

    def test_v10_vs_v11_ci_rule_severity_differs(self):
        # v1.0 的 CI 规则为 warning，v1.1 为 invalid
        v10_ci = next(c for c in V10.checks if c.code == "POTENCY_CI_WIDTH")
        v11_ci = next(c for c in V11.checks if c.code == "POTENCY_CI_WIDTH")
        self.assertEqual(v10_ci.severity, "warning")
        self.assertEqual(v11_ci.severity, "invalid")

    def test_evaluation_is_pure_and_reproducible(self):
        f = fit_dict(0.9)
        a = evaluate_plate(f, V11).to_dict()
        b = evaluate_plate(f, V11).to_dict()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
