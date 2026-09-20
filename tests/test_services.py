"""服务层工作流测试：覆盖题目场景 —— 离散读数板、排除孔审计、
首轮锁定、复测会审、组合结论、重复导入识别与全程追溯。"""
import json

import pytest

from potency.errors import DuplicatePlateError, NotFoundError, ValidationError, WorkflowError
from potency.services import (
    BATCH_CONCLUDED,
    BATCH_FIRST_ROUND_LOCKED,
    BATCH_OPEN,
    ROLE_ANALYST,
    ROLE_QA,
    ROLE_SUPERVISOR,
)
from tests.conftest import import_and_analyze, make_plate


class TestImportAndDuplicate:
    def test_import_stores_hash_and_payload(self, service, batch):
        plate = service.import_plate(**make_plate())
        assert plate["content_hash"]
        assert len(plate["payload"]["wells"]) == 42

    def test_duplicate_import_rejected(self, service, batch):
        service.import_plate(**make_plate())
        with pytest.raises(DuplicatePlateError) as exc:
            service.import_plate(**make_plate(label="P-002"))  # 换标签仍是同一内容
        assert exc.value.existing_plate_id == 1

    def test_different_readings_not_duplicate(self, service, batch):
        service.import_plate(**make_plate())
        other = make_plate(label="P-002", potency=0.9)
        plate = service.import_plate(**other)
        assert plate["id"] == 2

    def test_unknown_batch_rejected(self, service):
        with pytest.raises(NotFoundError):
            service.import_plate(**make_plate(batch_code="NOPE"))

    def test_unknown_standard_rejected(self, service, batch):
        with pytest.raises(NotFoundError):
            service.import_plate(**make_plate(standard_lot="NOPE"))

    def test_bad_well_series_rejected(self, service, batch):
        plate = make_plate()
        plate["wells"][0]["series"] = "GHOST"
        with pytest.raises(ValidationError):
            service.import_plate(**plate)


class TestAnalysis:
    def test_valid_plate_all_rules_pass(self, service, batch):
        _, analysis = import_and_analyze(service, make_plate(potency=0.85))
        sample = analysis["result"]["samples"]["S1"]
        assert sample["valid"]
        assert analysis["result"]["fits"]["STD"]["status"] == "CONVERGED"
        assert sample["potency"]["potency_pct"] == pytest.approx(85.0, abs=2.0)

    def test_analysis_snapshots_ruleset(self, service, batch):
        _, analysis = import_and_analyze(service, make_plate())
        assert analysis["ruleset_snapshot"]["max_replicate_rsd_pct"] == 15.0
        assert analysis["fit_config_snapshot"]["max_iter"] == 200

    def test_nonparallel_sample_invalid(self, service, batch):
        _, analysis = import_and_analyze(service, make_plate(sample_B=2.4, noise=0.01))
        sample = analysis["result"]["samples"]["S1"]
        assert not sample["valid"]
        hits = {h["rule_id"]: h for h in sample["rule_hits"]}
        assert not hits["R06_PARALLELISM"]["passed"]

    def test_scattered_plate_fails_precision(self, service, batch):
        plate = make_plate()
        # 人为制造一个离散孔：读数放大 3 倍
        for w in plate["wells"]:
            if w["series"] == "S1" and w["level"] == 2 and w["response"] < 2.0:
                w["response"] = round(w["response"] * 3.0, 6)
                break
        _, analysis = import_and_analyze(service, plate)
        sample = analysis["result"]["samples"]["S1"]
        assert not sample["valid"]
        hits = {h["rule_id"]: h for h in sample["rule_hits"]}
        assert not hits["R04_PRECISION"]["passed"]

    def test_recompute_is_deterministic(self, service, batch):
        _, analysis = import_and_analyze(service, make_plate())
        check = service.recompute_analysis(analysis["id"])
        assert check["matches"]

    def test_determinations_created_and_superseded(self, service, batch):
        plate, _ = import_and_analyze(service, make_plate())
        service.analyze_plate(plate["id"], actor="analyst.chen")
        dets = service.list_determinations(batch["id"], include_superseded=True)
        assert len(dets) == 2
        assert sum(1 for d in dets if d["superseded"]) == 1
        current = service.list_determinations(batch["id"])
        assert len(current) == 1 and not current[0]["superseded"]


class TestExclusion:
    def _scattered_plate(self):
        plate = make_plate(potency=0.82)
        target = None
        for w in plate["wells"]:
            if w["series"] == "S1" and w["level"] == 2 and w["well"].endswith("6"):
                target = w
                break
        assert target is not None
        target["response"] = round(target["response"] * 3.0, 6)
        return plate, target["well"]

    def test_exclusion_records_operator_reason_and_delta(self, service, batch):
        plate_payload, bad_well = self._scattered_plate()
        plate, before_analysis = import_and_analyze(service, plate_payload)
        assert not before_analysis["result"]["samples"]["S1"]["valid"]

        exc = service.exclude_well(plate["id"], bad_well, operator="analyst.chen",
                                   reason="复孔离散超差，该孔读数疑为加样错误")
        assert exc["operator"] == "analyst.chen"
        assert exc["reason"].startswith("复孔离散")
        assert exc["before_analysis_id"] == before_analysis["id"]
        assert exc["after_analysis_id"] is not None
        # 前后效价差异已登记
        assert exc["before_potency"]["S1"] is not None
        assert exc["after_potency"]["S1"] is not None
        assert exc["delta_potency"]["S1"] == pytest.approx(
            exc["after_potency"]["S1"] - exc["before_potency"]["S1"]
        )
        # 排除后板转为有效
        after = service.get_analysis(exc["after_analysis_id"])
        assert after["result"]["samples"]["S1"]["valid"]
        assert bad_well in after["result"]["excluded_wells"]

    def test_exclusion_does_not_modify_raw_readings(self, service, batch):
        plate_payload, bad_well = self._scattered_plate()
        plate, _ = import_and_analyze(service, plate_payload)
        raw_before = json.dumps(service.get_plate(plate["id"])["payload"], sort_keys=True)
        service.exclude_well(plate["id"], bad_well, operator="analyst.chen", reason="离散")
        raw_after = json.dumps(service.get_plate(plate["id"])["payload"], sort_keys=True)
        assert raw_before == raw_after

    def test_double_exclusion_rejected(self, service, batch):
        plate_payload, bad_well = self._scattered_plate()
        plate, _ = import_and_analyze(service, plate_payload)
        service.exclude_well(plate["id"], bad_well, operator="a", reason="离散")
        with pytest.raises(WorkflowError):
            service.exclude_well(plate["id"], bad_well, operator="b", reason="重复")

    def test_unknown_well_rejected(self, service, batch):
        plate, _ = import_and_analyze(service, make_plate())
        with pytest.raises(NotFoundError):
            service.exclude_well(plate["id"], "Z99", operator="a", reason="不存在")

    def test_reason_required(self, service, batch):
        plate, _ = import_and_analyze(service, make_plate())
        with pytest.raises(ValidationError):
            service.exclude_well(plate["id"], "A1", operator="a", reason="  ")


class TestRetestWorkflow:
    def _locked_batch(self, service, batch):
        import_and_analyze(service, make_plate(potency=0.78))  # 首轮：低于放行限度
        return service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)

    def test_lock_requires_determination(self, service, batch):
        with pytest.raises(WorkflowError):
            service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)

    def test_lock_requires_role(self, service, batch):
        import_and_analyze(service, make_plate())
        with pytest.raises(WorkflowError):
            service.lock_first_round(batch["id"], actor="analyst.chen", role=ROLE_ANALYST)

    def test_retest_requires_lock_first(self, service, batch):
        import_and_analyze(service, make_plate())
        with pytest.raises(WorkflowError, match="首轮"):
            service.request_retest(batch["id"], actor="analyst.chen", role=ROLE_ANALYST,
                                   reason="首轮结果临近限度")

    def test_full_retest_flow(self, service, batch):
        first = self._locked_batch(service, batch)
        assert first["kind"] == "FIRST_ROUND"
        assert first["combined_potency_pct"] == pytest.approx(78.0, abs=2.0)
        assert first["outcome"] == "FAIL"  # 低于 80% 放行限度
        assert service.get_batch(batch["id"])["status"] == BATCH_FIRST_ROUND_LOCKED

        # 锁定后无申请不得导入新板
        with pytest.raises(WorkflowError):
            service.import_plate(**make_plate(label="P-RT", potency=0.86))

        req = service.request_retest(batch["id"], actor="analyst.chen", role=ROLE_ANALYST,
                                     reason="首轮效价 78% 低于放行限度，申请复测一板")
        assert req["status"] == "PENDING"

        # 未批准不能导入复测板
        with pytest.raises(WorkflowError):
            service.import_plate(**make_plate(label="P-RT", potency=0.86),
                                 retest_request_id=req["id"])

        # 本人不得批准本人申请
        with pytest.raises(WorkflowError):
            service.decide_retest(req["id"], actor="analyst.chen", role=ROLE_SUPERVISOR, approve=True)
        # 批准角色必须与申请角色不同
        with pytest.raises(WorkflowError):
            service.decide_retest(req["id"], actor="analyst.zhao", role=ROLE_ANALYST, approve=True)

        approved = service.decide_retest(req["id"], actor="sup.li", role=ROLE_SUPERVISOR, approve=True)
        assert approved["status"] == "APPROVED"
        assert approved["approved_by"] == "sup.li"

        # 批准后导入复测板
        rt_plate = service.import_plate(**make_plate(label="P-RT", potency=0.86),
                                        retest_request_id=req["id"])
        service.analyze_plate(rt_plate["id"], actor="analyst.chen")

        # 最终结论：纳入全部有效测定（首轮 + 复测），不允许挑选
        final = service.conclude_batch(batch["id"], actor="qa.wang", role=ROLE_QA)
        assert final["kind"] == "FINAL"
        assert final["n_determinations"] == 2
        assert len(final["determination_ids"]) == 2
        assert final["combined_potency_pct"] == pytest.approx(82.0, abs=2.5)
        assert final["outcome"] == "PASS"
        assert service.get_batch(batch["id"])["status"] == BATCH_CONCLUDED

        # 结论后批次封闭
        with pytest.raises(WorkflowError):
            service.import_plate(**make_plate(label="P-X", potency=0.9))

    def test_conclusion_includes_all_valid_determinations(self, service, batch):
        """机制性保证：最终结论的测定集合 == 当前全部有效测定。"""
        import_and_analyze(service, make_plate(potency=0.82))
        import_and_analyze(service, make_plate(label="P-2", potency=0.84))
        # 一块无效板（不平行）—— 不进入组合
        import_and_analyze(service, make_plate(label="P-3", potency=0.99, sample_B=2.5, noise=0.01))
        service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)
        final = service.conclude_batch(batch["id"], actor="qa.wang", role=ROLE_QA)
        valid_ids = {d["id"] for d in service.list_determinations(batch["id"]) if d["valid"]}
        assert set(final["determination_ids"]) == valid_ids
        assert final["n_determinations"] == 2

    def test_conclude_without_valid_determination_fails(self, service, batch):
        import_and_analyze(service, make_plate(sample_B=2.5, noise=0.01))  # 无效板
        service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)
        with pytest.raises(WorkflowError, match="有效测定"):
            service.conclude_batch(batch["id"], actor="qa.wang", role=ROLE_QA)

    def test_double_lock_rejected(self, service, batch):
        self._locked_batch(service, batch)
        with pytest.raises(WorkflowError):
            service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)

    def test_double_decision_rejected(self, service, batch):
        self._locked_batch(service, batch)
        req = service.request_retest(batch["id"], actor="analyst.chen", role=ROLE_ANALYST, reason="r")
        service.decide_retest(req["id"], actor="sup.li", role=ROLE_SUPERVISOR, approve=True)
        with pytest.raises(WorkflowError):
            service.decide_retest(req["id"], actor="qa.wang", role=ROLE_QA, approve=True)


class TestTraceability:
    def test_trace_chain_and_integrity(self, service, batch):
        plate, analysis = import_and_analyze(service, make_plate(potency=0.83))
        service.lock_first_round(batch["id"], actor="sup.li", role=ROLE_SUPERVISOR)
        service.conclude_batch(batch["id"], actor="qa.wang", role=ROLE_QA)

        trace = service.trace_batch(batch["id"])
        assert trace["integrity"]["all_raw_readings_intact"]
        assert trace["plates"][0]["hash_verified"]
        # 结论 → 测定 → 分析 → 板 → 原始读数 全链路可达
        final = [c for c in trace["conclusions"] if c["kind"] == "FINAL"][0]
        det_ids = set(final["determination_ids"])
        dets = {d["id"]: d for d in trace["determinations"]}
        assert det_ids <= set(dets)
        for did in det_ids:
            ana = next(a for a in [analysis] if a["id"] == dets[did]["analysis_id"])
            assert ana["plate_id"] == plate["id"]
        assert trace["plates"][0]["standard_lot"] == "STD-01"

    def test_tampered_raw_reading_detected(self, service, batch):
        plate, _ = import_and_analyze(service, make_plate())
        # 模拟有人直接改库里的原始读数
        payload = service.get_plate(plate["id"])["payload"]
        payload["wells"][0]["response"] = 9.999
        service.store.update("plates", plate["id"], {"payload": payload})
        trace = service.trace_batch(batch["id"])
        assert not trace["plates"][0]["hash_verified"]
        assert not trace["integrity"]["all_raw_readings_intact"]


class TestRulesetVersioning:
    def test_historical_analysis_uses_its_own_ruleset(self, service, batch):
        """方法版本升级后，历史分析仍按当时规则集复算。"""
        plate, analysis = import_and_analyze(service, make_plate(noise=0.06))
        assert analysis["result"]["samples"]["S1"]["valid"]
        # 新建更严格的方法版本（不影响已完成的分析）
        service.create_method_version(code="CYTO-POT", name="细胞法效价测定", version="2.0",
                                      ruleset={"max_replicate_rsd_pct": 2.0}, actor="qa.wang")
        check = service.recompute_analysis(analysis["id"])
        assert check["matches"]  # 仍按 v1.0 规则集复算，结果一致
        assert check["recomputed"]["samples"]["S1"]["valid"]
