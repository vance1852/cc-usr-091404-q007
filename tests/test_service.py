"""服务层工作流测试：导入去重、版本、排孔审计、规则时点、锁定、
复测审批、批次组合与追溯。
"""

import json
import sqlite3
import unittest
from datetime import date, datetime, timezone

from tests.conftest import ServiceTestCase
from tests.datafactory import make_wells, plate_payload
from potency.service import DuplicateImport, ServiceError


def outlier_payload(code: str, well: str = "E09", shift: float = 0.15,
                    sample_ec50: float = 1.0) -> dict:
    """在样品 EC50 附近剂量孔注入离群读数（默认使 80.8%→77.1%）。"""
    wells = make_wells(sample_ec50=sample_ec50)
    for w in wells:
        if w["well"] == well:
            w["reading"] = round(w["reading"] + shift, 6)
    return plate_payload(code, wells)


class TestImportAndImmutability(ServiceTestCase):
    def test_import_preserves_metadata_and_readings(self):
        plate = self.import_good_plate("PL-1")
        self.assertEqual(plate["method"]["version"], "2.3")
        self.assertEqual(plate["standard_lot"], "STD-LOT-2026-01")
        self.assertEqual(len(plate["readings"]), 112)  # 4行×7剂量×2重复×2角色
        self.assertEqual(len(plate["dilution"]["standard"]), 7)
        self.assertEqual(plate["status"], "imported")
        self.assertRegex(plate["content_hash"], r"^[0-9a-f]{64}$")

    def test_duplicate_import_detected_even_with_new_code(self):
        self.import_good_plate("PL-1")
        wells = make_wells()
        again = plate_payload("PL-1-DUP", wells)  # 仅板号/文件名不同
        with self.assertRaises(DuplicateImport) as cm:
            self.service.import_plate(again, "analyst.li")
        self.assertEqual(cm.exception.code, "DUPLICATE_IMPORT")
        self.assertEqual(cm.exception.existing_plate_code, "PL-1")

    def test_content_hash_ignores_source_name_but_catches_reading_change(self):
        p1 = plate_payload("PL-A", make_wells())
        p1["source_name"] = "machine1.csv"
        p2 = plate_payload("PL-B", make_wells())
        p2["source_name"] = "machine2.csv"
        self.service.import_plate(p1, "analyst.li")
        with self.assertRaises(DuplicateImport):
            self.service.import_plate(p2, "analyst.li")
        # 改动任一读数即视为新板
        changed = plate_payload("PL-C", make_wells())
        changed["wells"][0]["reading"] += 0.001
        plate_c = self.service.import_plate(changed, "analyst.li")
        self.assertIsNotNone(plate_c["id"])

    def test_raw_readings_cannot_be_modified_or_deleted(self):
        plate = self.import_good_plate("PL-IMM")
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "UPDATE plate_readings SET reading=9.9 WHERE plate_id=?", (plate["id"],)
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "DELETE FROM plate_readings WHERE plate_id=?", (plate["id"],)
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("DELETE FROM raw_imports WHERE plate_id=?", (plate["id"],))

    def test_import_requires_analyst_role(self):
        payload = plate_payload("PL-X", make_wells())
        with self.assertRaises(ServiceError) as cm:
            self.service.import_plate(payload, "qa.zhao")
        self.assertEqual(cm.exception.code, "FORBIDDEN_ROLE")

    def test_bad_payload_rejected(self):
        payload = plate_payload("PL-Y", make_wells())
        payload["wells"][0]["dose"] = -1
        with self.assertRaises(ServiceError):
            self.service.import_plate(payload, "analyst.li")


class TestAnalysisVersions(ServiceTestCase):
    def test_first_analysis_is_version_1_and_valid(self):
        plate = self.import_good_plate("PL-V", sample_ec50=0.85)
        a = self.service.create_analysis(plate["id"], "analyst.li")
        self.assertEqual(a["version_no"], 1)
        self.assertEqual(a["ruleset"]["version"], "1.1")
        self.assertTrue(a["evaluation"]["valid"])
        self.assertTrue(a["evaluation"]["release_passed"])

    def test_versions_are_append_only(self):
        plate = self.import_good_plate("PL-AP", sample_ec50=0.85)
        self.service.create_analysis(plate["id"], "analyst.li", note="v1")
        self.service.create_analysis(plate["id"], "analyst.li", note="v2")
        versions = self.service.list_plate_analyses(plate["id"])
        self.assertEqual([v["version_no"] for v in versions], [1, 2])
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("UPDATE analyses SET note='hack' WHERE id=?",
                              (versions[0]["id"],))

    def test_exclusion_records_before_after_delta(self):
        # 离群板首轮失败（77.1%），排除 E09 后回升（80.8%）
        plate = self.service.import_plate(outlier_payload("PL-EX"), "analyst.li")
        bad = self.service.create_analysis(plate["id"], "analyst.li",
                                           note="首轮：存在离群孔")
        pot_before = bad["fit"]["relative_potency"]
        self.assertLess(pot_before, 80.0)  # 低于放行限

        fixed = self.service.create_analysis(
            plate["id"], "analyst.li",
            exclusions=[{
                "well": "E09", "operator": "analyst.li",
                "reason": "移液异常（复核签认）",
            }],
            note="排除离群孔后重算",
        )
        self.assertEqual(len(fixed["exclusions"]), 1)
        rec = fixed["exclusions"][0]
        self.assertEqual(rec["operator"], "analyst.li")
        self.assertEqual(rec["reason"], "移液异常（复核签认）")
        self.assertAlmostEqual(rec["potency_before"], pot_before, places=8)
        self.assertGreater(rec["potency_after"], rec["potency_before"])
        self.assertGreater(rec["delta_potency_pct"], 0.0)
        # 原始读数仍保留被排除孔的值
        self.assertIsNotNone(rec["reading"])

    def test_exclusion_requires_reason_and_operator(self):
        plate = self.service.import_plate(outlier_payload("PL-EX2"), "analyst.li")
        with self.assertRaises(ServiceError):
            self.service.create_analysis(
                plate["id"], "analyst.li",
                exclusions=[{"well": "E09", "operator": "", "reason": ""}],
            )

    def test_nonparallel_plate_evaluated_invalid(self):
        wells = make_wells(sample_ec50=0.85)
        for w in wells:
            if w["role"] == "sample":
                d = w["dose"]
                w["reading"] = round(w["reading"] + 0.25 * d**2 / (d**2 + 1), 6)
        plate = self.service.import_plate(plate_payload("PL-NP", wells), "analyst.li")
        a = self.service.create_analysis(plate["id"], "analyst.li")
        self.assertFalse(a["evaluation"]["valid"])
        codes = {h["code"] for h in a["evaluation"]["hits"]}
        self.assertIn("PARALLELISM_P", codes)

    def test_underdetermined_plate_marked_invalid_not_crash(self):
        # 仅 3 个剂量水平：4PL 无法拟合 → 收敛/剂量规则判废，不能抛异常
        from potency.fitting import four_pl

        wells = []
        doses = [0.25, 1.0, 4.0]
        for role, ec50 in (("standard", 0.8), ("sample", 1.0)):
            rows = "ABCD" if role == "standard" else "EFGH"
            for r, row in enumerate(rows):
                for di, d in enumerate(doses):
                    wells.append({
                        "well": f"{row}{di+1:02d}", "role": role, "dose": d,
                        "reading": round(four_pl(d, 0.2, 1.1, ec50, 1.5), 6),
                    })
        plate = self.service.import_plate(
            plate_payload("PL-FEW", wells), "analyst.li"
        )
        a = self.service.create_analysis(plate["id"], "analyst.li")
        self.assertFalse(a["evaluation"]["valid"])
        self.assertFalse(a["fit"]["converged"])
        codes = {h["code"] for h in a["evaluation"]["hits"]}
        self.assertIn("FIT_CONVERGED", codes)


class TestRulesEffectiveAtTime(ServiceTestCase):
    def test_analysis_uses_ruleset_effective_on_lock_era(self):
        plate = self.import_good_plate("PL-R")
        # 2025 年分析 → v1.0
        self.set_clock(datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc))
        old = self.service.create_analysis(plate["id"], "analyst.li")
        self.assertEqual(old["ruleset"]["version"], "1.0")
        # 推进到 2026 年再分析 → v1.1
        self.set_clock(datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc))
        new = self.service.create_analysis(plate["id"], "analyst.li")
        self.assertEqual(new["ruleset"]["version"], "1.1")

    def test_historical_recompute_matches_stored(self):
        # 旧版本结论必须可按旧规则复算
        plate = self.import_good_plate("PL-H")
        self.set_clock(datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc))
        old = self.service.create_analysis(plate["id"], "analyst.li")
        stored_eval = old["evaluation"]
        from potency.rules import evaluate_plate, ruleset_effective_on, DEFAULT_RULES
        rs = ruleset_effective_on(DEFAULT_RULES, date(2025, 6, 1))
        recomputed = evaluate_plate(old["fit"], rs).to_dict()
        self.assertEqual(recomputed, stored_eval)


from datetime import date  # noqa: E402  (供上面测试使用)


class TestLocking(ServiceTestCase):
    def test_lock_freezes_plate(self):
        plate = self.import_good_plate("PL-L")
        self.service.create_analysis(plate["id"], "analyst.li")
        locked = self.service.lock_first_round(plate["id"], "supervisor.wang")
        self.assertEqual(locked["status"], "locked")
        with self.assertRaises(ServiceError) as cm:
            self.service.create_analysis(locked["id"], "analyst.li")
        self.assertEqual(cm.exception.code, "PLATE_LOCKED")

    def test_invalid_plate_locks_as_invalid(self):
        wells = make_wells(sample_ec50=0.85)
        for w in wells:
            if w["role"] == "sample":
                d = w["dose"]
                w["reading"] = round(w["reading"] + 0.25 * d**2 / (d**2 + 1), 6)
        plate = self.service.import_plate(plate_payload("PL-LI", wells), "analyst.li")
        self.service.create_analysis(plate["id"], "analyst.li")
        locked = self.service.lock_first_round(plate["id"], "supervisor.wang")
        self.assertEqual(locked["status"], "invalid_locked")

    def test_analyst_cannot_lock(self):
        plate = self.import_good_plate("PL-LC")
        self.service.create_analysis(plate["id"], "analyst.li")
        with self.assertRaises(ServiceError) as cm:
            self.service.lock_first_round(plate["id"], "analyst.li")
        self.assertEqual(cm.exception.code, "FORBIDDEN_ROLE")

    def test_cannot_lock_unanalyzed_plate(self):
        plate = self.import_good_plate("PL-LU")
        with self.assertRaises(ServiceError):
            self.service.lock_first_round(plate["id"], "supervisor.wang")


class TestRetestWorkflow(ServiceTestCase):
    def _locked_marginal(self, code="PL-M"):
        # 合格但低于放行限的板（77.25%）
        plate = self.import_good_plate(code, sample_ec50=1.05)
        self.service.create_analysis(plate["id"], "analyst.li")
        return self.service.lock_first_round(plate["id"], "supervisor.wang")

    def test_retest_requires_locked_plate(self):
        plate = self.import_good_plate("PL-RR")
        with self.assertRaises(ServiceError) as cm:
            self.service.create_retest_request(plate["id"], "analyst.li", "原因")
        self.assertEqual(cm.exception.code, "PLATE_NOT_LOCKED")

    def test_retest_requires_two_distinct_roles(self):
        plate = self._locked_marginal()
        req = self.service.create_retest_request(
            plate["id"], "analyst.li", "首读 77%，怀疑边缘漂移"
        )
        self.assertEqual(req["status"], "pending")
        # 仅主管批准还不够
        req = self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        self.assertEqual(req["status"], "pending")
        # QA 批准后才通过
        req = self.service.decide_retest(req["id"], "qa.zhao", "approved")
        self.assertEqual(req["status"], "approved")
        roles = {a["approver_role"] for a in req["approvals"]}
        self.assertEqual(roles, {"SUPERVISOR", "QA"})

    def test_applicant_cannot_self_approve(self):
        plate = self._locked_marginal("PL-SA")
        req = self.service.create_retest_request(
            plate["id"], "supervisor.wang", "主管自己发起的申请"
        )
        with self.assertRaises(ServiceError) as cm:
            self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        self.assertEqual(cm.exception.code, "SELF_APPROVAL")

    def test_qa_rejection_closes_request(self):
        plate = self._locked_marginal("PL-RJ")
        req = self.service.create_retest_request(plate["id"], "analyst.li", "x")
        req = self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        req = self.service.decide_retest(req["id"], "qa.zhao", "rejected",
                                         comment="证据不足")
        self.assertEqual(req["status"], "rejected")

    def test_cannot_import_retest_plate_without_approved_request(self):
        plate = self._locked_marginal("PL-RA")
        req = self.service.create_retest_request(plate["id"], "analyst.li", "x")
        bad = plate_payload("PL-RT1", make_wells(sample_ec50=0.85),
                            retest_request_id=req["id"])
        with self.assertRaises(ServiceError) as cm:
            self.service.import_plate(bad, "analyst.li")
        self.assertEqual(cm.exception.code, "REQUEST_NOT_APPROVED")

    def test_approved_retest_plate_shares_lineage(self):
        plate = self._locked_marginal("PL-LN")
        req = self.service.create_retest_request(plate["id"], "analyst.li", "x")
        self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        self.service.decide_retest(req["id"], "qa.zhao", "approved")
        retest = self.service.import_plate(
            plate_payload("PL-LN-R1", make_wells(sample_ec50=0.85),
                          retest_request_id=req["id"]),
            "analyst.li",
        )
        original = self.service.get_plate(plate["id"])
        self.assertEqual(retest["lineage_key"], original["lineage_key"])
        self.assertEqual(retest["retest_of_request_id"], req["id"])


class TestBatchConclusion(ServiceTestCase):
    def _full_lineage(self, base_code, retest_code, base_ec50=1.05,
                      retest_ec50=0.85):
        """造一条：首轮锁定（失败边缘）→ 批准复测 → 复测板（合格）。"""
        plate = self.import_good_plate(base_code, sample_ec50=base_ec50)
        self.service.create_analysis(plate["id"], "analyst.li")
        self.service.lock_first_round(plate["id"], "supervisor.wang")
        req = self.service.create_retest_request(plate["id"], "analyst.li", "边缘")
        self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        self.service.decide_retest(req["id"], "qa.zhao", "approved")
        retest = self.service.import_plate(
            plate_payload(retest_code, make_wells(sample_ec50=retest_ec50),
                          retest_request_id=req["id"]),
            "analyst.li",
        )
        self.service.create_analysis(retest["id"], "analyst.li")
        self.service.lock_first_round(retest["id"], "supervisor.wang")
        return plate["id"], retest["id"]

    def test_conclusion_includes_all_plates_in_lineage(self):
        p1, p2 = self._full_lineage("PL-B1", "PL-B1-R1")
        batch = self.service.create_batch("B-1", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p1, "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p2, "analyst.li")
        result = self.service.conclude_batch(batch["id"], "qa.zhao")
        self.assertEqual(result["status"], "concluded")
        contrib = result["conclusion"]["contributions"]
        self.assertEqual(len(contrib), 2)

    def test_cannot_cherry_pick_only_favorable_retest(self):
        p1, p2 = self._full_lineage("PL-C1", "PL-C1-R1")
        batch = self.service.create_batch("B-2", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        # 只加复测好板，故意漏掉首轮失败板
        self.service.add_plate_to_batch(batch["id"], p2, "analyst.li")
        with self.assertRaises(ServiceError) as cm:
            self.service.conclude_batch(batch["id"], "qa.zhao")
        self.assertEqual(cm.exception.code, "LINEAGE_INCOMPLETE")

    def test_invalid_plate_makes_batch_fail_even_with_good_retest(self):
        # 首轮板非平行（无效），复测合格：组合仍不放行
        wells = make_wells(sample_ec50=0.85)
        for w in wells:
            if w["role"] == "sample":
                d = w["dose"]
                w["reading"] = round(w["reading"] + 0.25 * d**2 / (d**2 + 1), 6)
        plate = self.service.import_plate(plate_payload("PL-I1", wells), "analyst.li")
        self.service.create_analysis(plate["id"], "analyst.li")
        self.service.lock_first_round(plate["id"], "supervisor.wang")
        req = self.service.create_retest_request(plate["id"], "analyst.li", "非平行")
        self.service.decide_retest(req["id"], "supervisor.wang", "approved")
        self.service.decide_retest(req["id"], "qa.zhao", "approved")
        retest = self.service.import_plate(
            plate_payload("PL-I1-R1", make_wells(sample_ec50=0.85),
                          retest_request_id=req["id"]),
            "analyst.li",
        )
        self.service.create_analysis(retest["id"], "analyst.li")
        self.service.lock_first_round(retest["id"], "supervisor.wang")

        batch = self.service.create_batch("B-3", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], plate["id"], "analyst.li")
        self.service.add_plate_to_batch(batch["id"], retest["id"], "analyst.li")
        result = self.service.conclude_batch(batch["id"], "qa.zhao")
        self.assertFalse(result["conclusion"]["released"])
        self.assertFalse(result["conclusion"]["all_plates_valid"])

    def test_pending_retest_blocks_conclusion(self):
        p1, p2 = self._full_lineage("PL-P1", "PL-P1-R1")
        # 再造一条只有待审批申请的谱系（不同 EC50 → 不同内容哈希）
        plate2 = self.import_good_plate("PL-P2", sample_ec50=1.04)
        self.service.create_analysis(plate2["id"], "analyst.li")
        self.service.lock_first_round(plate2["id"], "supervisor.wang")
        self.service.create_retest_request(plate2["id"], "analyst.li", "等待中")
        batch = self.service.create_batch("B-4", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], plate2["id"], "analyst.li")
        with self.assertRaises(ServiceError) as cm:
            self.service.conclude_batch(batch["id"], "qa.zhao")
        self.assertEqual(cm.exception.code, "PENDING_RETEST")

    def test_geometric_mean_strategy(self):
        p1, p2 = self._full_lineage("PL-G1", "PL-G1-R1")
        batch = self.service.create_batch("B-5", "产品X", "GMEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p1, "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p2, "analyst.li")
        result = self.service.conclude_batch(batch["id"], "qa.zhao")
        self.assertEqual(result["conclusion"]["aggregation"], "geometric_mean")

    def test_analyst_cannot_conclude(self):
        p1, p2 = self._full_lineage("PL-Q1", "PL-Q1-R1")
        batch = self.service.create_batch("B-6", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p1, "analyst.li")
        self.service.add_plate_to_batch(batch["id"], p2, "analyst.li")
        with self.assertRaises(ServiceError) as cm:
            self.service.conclude_batch(batch["id"], "analyst.li")
        self.assertEqual(cm.exception.code, "FORBIDDEN_ROLE")


class TestComparisonAndTrace(ServiceTestCase):
    def test_compare_versions_shows_curve_and_rule_diff(self):
        plate = self.service.import_plate(outlier_payload("PL-CMP"), "analyst.li")
        v1 = self.service.create_analysis(plate["id"], "analyst.li")
        v2 = self.service.create_analysis(
            plate["id"], "analyst.li",
            exclusions=[{"well": "E09", "operator": "analyst.li",
                         "reason": "移液异常"}],
        )
        cmp = self.service.compare_analyses(v1["id"], v2["id"])
        self.assertEqual(cmp["exclusions_added_in_b"], ["E09"])
        self.assertNotEqual(cmp["potency"]["a_percent"],
                            cmp["potency"]["b_percent"])
        # 曲线参数两侧都存在
        self.assertIn("B", cmp["curves"]["a"]["standard"])

    def test_trace_conclusion_back_to_unmodified_readings(self):
        plate = self.import_good_plate("PL-TR", sample_ec50=0.85)
        self.service.create_analysis(plate["id"], "analyst.li")
        self.service.lock_first_round(plate["id"], "supervisor.wang")
        batch = self.service.create_batch("B-T", "产品X", "MEAN_ALL_VALID",
                                          "analyst.li")
        self.service.add_plate_to_batch(batch["id"], plate["id"], "analyst.li")
        self.service.conclude_batch(batch["id"], "qa.zhao")
        sources = self.service.trace_batch_readings(batch["id"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(sources[0]["readings"]), 112)
        self.assertIsNotNone(sources[0]["raw_import"]["content_hash"])
        # 追溯值与原始导入载荷一致
        raw = self.conn.execute(
            "SELECT payload_json FROM raw_imports WHERE plate_id=?", (plate["id"],)
        ).fetchone()
        payload = json.loads(raw["payload_json"])
        self.assertEqual(len(payload["wells"]), len(sources[0]["readings"]))


if __name__ == "__main__":
    unittest.main()
