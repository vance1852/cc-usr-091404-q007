"""规则集评估测试：每条规则的触发与版本化差异。"""
from potency.rules import DEFAULT_RULESET, all_passed, evaluate_ruleset, merged_ruleset

GOOD_CTX = {
    "std_converged": True, "smp_converged": True,
    "std_status": "CONVERGED", "smp_status": "CONVERGED",
    "span_std": 1.9, "span_smp": 1.85,
    "r2_std": 0.999, "r2_smp": 0.998,
    "rsd_std": 3.0, "rsd_smp": 4.0,
    "valid_levels_std": 4, "valid_levels_smp": 4,
    "parallelism_p": 0.42,
    "potency_estimable": True,
}


def hits_by_id(hits):
    return {h["rule_id"]: h for h in hits}


class TestAllPass:
    def test_good_context_valid(self):
        hits = evaluate_ruleset(None, GOOD_CTX)
        assert len(hits) == 7
        assert all_passed(hits)


class TestIndividualRules:
    def test_convergence_failure(self):
        ctx = {**GOOD_CTX, "smp_converged": False, "smp_status": "MAX_ITERATIONS"}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R01_CONVERGENCE"]["passed"]
        assert not all_passed(list(hits.values()))

    def test_span_failure(self):
        ctx = {**GOOD_CTX, "span_smp": 0.01}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R02_RESPONSE_SPAN"]["passed"]
        assert hits["R02_RESPONSE_SPAN"]["value"] == 0.01

    def test_r_squared_failure(self):
        ctx = {**GOOD_CTX, "r2_std": 0.90}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R03_R_SQUARED"]["passed"]

    def test_precision_failure(self):
        ctx = {**GOOD_CTX, "rsd_smp": 22.5}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R04_PRECISION"]["passed"]
        assert hits["R04_PRECISION"]["value"] == 22.5

    def test_valid_range_failure(self):
        ctx = {**GOOD_CTX, "valid_levels_smp": 2}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R05_VALID_RANGE"]["passed"]

    def test_parallelism_failure(self):
        ctx = {**GOOD_CTX, "parallelism_p": 0.001}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R06_PARALLELISM"]["passed"]

    def test_parallelism_none_fails(self):
        ctx = {**GOOD_CTX, "parallelism_p": None}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R06_PARALLELISM"]["passed"]

    def test_not_estimable(self):
        ctx = {**GOOD_CTX, "potency_estimable": False}
        hits = hits_by_id(evaluate_ruleset(None, ctx))
        assert not hits["R07_ESTIMABLE"]["passed"]


class TestVersionedRulesets:
    def test_same_data_different_verdict(self):
        """同一数据在不同版本规则集下判定不同 —— 复算必须用当时生效的规则集。"""
        ctx = {**GOOD_CTX, "rsd_smp": 18.0}
        strict = merged_ruleset({"max_replicate_rsd_pct": 15.0})
        lenient = merged_ruleset({"max_replicate_rsd_pct": 20.0})
        assert not all_passed(evaluate_ruleset(strict, ctx))
        assert all_passed(evaluate_ruleset(lenient, ctx))

    def test_merge_defaults(self):
        rs = merged_ruleset({"min_r_squared": 0.95})
        assert rs["min_r_squared"] == 0.95
        assert rs["max_replicate_rsd_pct"] == DEFAULT_RULESET["max_replicate_rsd_pct"]
