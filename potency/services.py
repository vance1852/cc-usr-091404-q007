"""业务服务层：主数据、板导入、分析计算、排除孔审计、复测会审、批次结论与追溯。

关键控制：
- 原始读数导入后不可修改；内容哈希用于识别重复导入与完整性核验；
- 每次分析生成新的不可变版本，快照当时生效的规则集与拟合参数；
- 排除孔必须登记操作者与理由，并自动记录排除前后的效价差异；
- 首轮结论锁定后才可申请复测，批准人必须与申请人不同人且不同角色；
- 最终结论按预先配置的组合策略纳入所有有效测定，没有挑选入口。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable

from .combine import combine_assay
from .errors import DuplicatePlateError, NotFoundError, ValidationError, WorkflowError
from .fitting import fit_4pl
from .metrics import (
    group_by_level,
    parallelism_test,
    relative_potency,
    replicate_rsd,
    valid_levels,
)
from .rules import all_passed, evaluate_ruleset, merged_ruleset
from .storage import Store, canonical_json, sanitize

ROLE_ANALYST = "ANALYST"
ROLE_SUPERVISOR = "SUPERVISOR"
ROLE_QA = "QA"
APPROVER_ROLES = (ROLE_SUPERVISOR, ROLE_QA)

BATCH_OPEN = "OPEN"
BATCH_FIRST_ROUND_LOCKED = "FIRST_ROUND_LOCKED"
BATCH_CONCLUDED = "CONCLUDED"

DEFAULT_FIT_CONFIG = {"max_iter": 200, "tolerance": 1e-12}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def plate_content_hash(core: dict) -> str:
    """板内容哈希：覆盖批号、标准品、系列定义与全部原始读数（不含标签等元数据）。"""
    return hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()


class PotencyService:
    """会审系统核心服务。clock 可注入以保证测试确定性。"""

    def __init__(self, db_path: str = ":memory:", clock: Callable[[], str] | None = None):
        self.store = Store(db_path)
        self.clock = clock or _utcnow

    # ------------------------------------------------------------------ 工具
    def _now(self) -> str:
        return self.clock()

    def _audit(self, entity: str, entity_id: int, action: str, actor: str, detail: Any = None) -> None:
        self.store.insert("audit_log", {
            "entity": entity, "entity_id": entity_id, "action": action,
            "actor": actor, "detail": detail, "created_at": self._now(),
        })

    @staticmethod
    def _require_role(role: str, allowed: tuple[str, ...], action: str) -> None:
        if role not in allowed:
            raise WorkflowError(f"{action} 需要角色 {'/'.join(allowed)}，当前角色为 {role}")

    def _get_or_404(self, table: str, row_id: int, label: str) -> dict:
        row = self.store.by_id(table, row_id)
        if row is None:
            raise NotFoundError(f"{label} 不存在: id={row_id}")
        return row

    # ------------------------------------------------------------------ 主数据
    def create_method_version(
        self,
        code: str,
        name: str,
        version: str,
        fit_config: dict | None = None,
        ruleset: dict | None = None,
        combination_strategy: str = "mean",
        release_low_pct: float = 80.0,
        release_high_pct: float = 125.0,
        actor: str = "system",
    ) -> dict:
        """登记实验方法版本：拟合参数、规则集、组合策略、放行限度一并固化。"""
        if self.store.one("method_versions", "code = ? AND version = ?", (code, version)):
            raise ValidationError(f"方法版本已存在: {code} v{version}")
        cfg = dict(DEFAULT_FIT_CONFIG)
        if fit_config:
            cfg.update(fit_config)
        row_id = self.store.insert("method_versions", {
            "code": code, "name": name, "version": version,
            "fit_config": cfg, "ruleset": merged_ruleset(ruleset),
            "combination_strategy": combination_strategy,
            "release_low_pct": float(release_low_pct),
            "release_high_pct": float(release_high_pct),
            "created_at": self._now(),
        })
        self._audit("method_versions", row_id, "CREATE", actor, {"code": code, "version": version})
        return self.store.by_id("method_versions", row_id)

    def get_method_version(self, method_id: int) -> dict:
        return self._get_or_404("method_versions", method_id, "方法版本")

    def list_method_versions(self) -> list[dict]:
        return self.store.all("method_versions")

    def create_standard(self, lot: str, assigned_potency: float = 1.0, unit: str = "U/mL", actor: str = "system") -> dict:
        if self.store.one("standards", "lot = ?", (lot,)):
            raise ValidationError(f"标准品批号已存在: {lot}")
        row_id = self.store.insert("standards", {
            "lot": lot, "assigned_potency": float(assigned_potency),
            "unit": unit, "created_at": self._now(),
        })
        self._audit("standards", row_id, "CREATE", actor, {"lot": lot})
        return self.store.by_id("standards", row_id)

    def get_standard(self, standard_id: int) -> dict:
        return self._get_or_404("standards", standard_id, "标准品")

    def create_batch(self, code: str, method_version_id: int, actor: str = "system") -> dict:
        """新建待检批次，绑定创建时生效的方法版本。"""
        self.get_method_version(method_version_id)
        if self.store.one("batches", "code = ?", (code,)):
            raise ValidationError(f"批次已存在: {code}")
        row_id = self.store.insert("batches", {
            "code": code, "method_version_id": method_version_id,
            "status": BATCH_OPEN, "created_at": self._now(),
        })
        self._audit("batches", row_id, "CREATE", actor, {"code": code})
        return self.store.by_id("batches", row_id)

    def get_batch(self, batch_id: int) -> dict:
        return self._get_or_404("batches", batch_id, "批次")

    # ------------------------------------------------------------------ 板导入
    @staticmethod
    def _normalize_payload(
        batch_code: str, standard_lot: str, series: dict, wells: list[dict]
    ) -> dict:
        """规范化板载荷：排序系列与孔位，保证哈希与分析顺序确定。"""
        norm_series = {}
        for name in sorted(series.keys()):
            sdef = series[name]
            kind = sdef.get("kind")
            if kind not in ("standard", "sample"):
                raise ValidationError(f"系列 {name} 的 kind 必须为 standard/sample")
            concs = [float(c) for c in sdef["concentrations"]]
            if not concs or any(c <= 0 for c in concs):
                raise ValidationError(f"系列 {name} 的标称浓度必须为正数")
            norm_series[name] = {"kind": kind, "concentrations": concs}
        norm_wells = []
        seen = set()
        for w in wells:
            key = (w["series"], int(w["level"]), w["well"])
            if w["series"] not in norm_series:
                raise ValidationError(f"孔 {w['well']} 引用了未定义的系列 {w['series']}")
            if not 0 <= int(w["level"]) < len(norm_series[w["series"]]["concentrations"]):
                raise ValidationError(f"孔 {w['well']} 的稀释水平越界: {w['level']}")
            if w["well"] in seen:
                raise ValidationError(f"孔位重复: {w['well']}")
            seen.add(w["well"])
            norm_wells.append({
                "well": str(w["well"]), "series": w["series"],
                "level": int(w["level"]), "response": float(w["response"]),
            })
        norm_wells.sort(key=lambda d: (d["series"], d["level"], d["well"]))
        if not any(s["kind"] == "standard" for s in norm_series.values()):
            raise ValidationError("板必须包含一个 standard 系列")
        if not any(s["kind"] == "sample" for s in norm_series.values()):
            raise ValidationError("板必须包含至少一个 sample 系列")
        return {
            "batch_code": batch_code,
            "standard_lot": standard_lot,
            "series": norm_series,
            "wells": norm_wells,
        }

    def import_plate(
        self,
        batch_code: str,
        plate_label: str,
        standard_lot: str,
        series: dict,
        wells: list[dict],
        imported_by: str,
        retest_request_id: int | None = None,
    ) -> dict:
        """导入检测板。内容哈希一致的重复导入会被拒绝并指向已存在的板。"""
        batch = self.store.one("batches", "code = ?", (batch_code,))
        if batch is None:
            raise NotFoundError(f"批次不存在: {batch_code}")
        if batch["status"] == BATCH_CONCLUDED:
            raise WorkflowError("批次已出最终结论，禁止再导入板")
        standard = self.store.one("standards", "lot = ?", (standard_lot,))
        if standard is None:
            raise NotFoundError(f"标准品批号不存在: {standard_lot}")
        if batch["status"] == BATCH_FIRST_ROUND_LOCKED:
            # 首轮锁定后只允许凭已批准的复测申请导入
            if retest_request_id is None:
                raise WorkflowError("首轮结论已锁定，导入复测板必须关联已批准的复测申请")
            req = self._get_or_404("retest_requests", retest_request_id, "复测申请")
            if req["batch_id"] != batch["id"]:
                raise WorkflowError("复测申请不属于该批次")
            if req["status"] != "APPROVED":
                raise WorkflowError("复测申请尚未批准，不能导入复测板")
        payload = self._normalize_payload(batch_code, standard_lot, series, wells)
        content_hash = plate_content_hash(payload)
        existing = self.store.one("plates", "content_hash = ?", (content_hash,))
        if existing is not None:
            raise DuplicatePlateError(
                f"检测到重复导入：内容与板 #{existing['id']}（{existing['plate_label']}）完全一致",
                existing_plate_id=existing["id"],
            )
        row_id = self.store.insert("plates", {
            "batch_id": batch["id"], "method_version_id": batch["method_version_id"],
            "standard_id": standard["id"], "plate_label": plate_label,
            "content_hash": content_hash, "payload": payload,
            "imported_by": imported_by, "imported_at": self._now(),
            "retest_request_id": retest_request_id,
        })
        self._audit("plates", row_id, "IMPORT", imported_by, {
            "batch_code": batch_code, "plate_label": plate_label, "content_hash": content_hash,
        })
        return self.store.by_id("plates", row_id)

    def get_plate(self, plate_id: int) -> dict:
        return self._get_or_404("plates", plate_id, "板")

    def find_duplicate(self, content_hash: str) -> dict | None:
        return self.store.one("plates", "content_hash = ?", (content_hash,))

    def list_plates(self, batch_id: int) -> list[dict]:
        return self.store.all("plates", "batch_id = ?", (batch_id,))

    # ------------------------------------------------------------------ 分析计算
    @staticmethod
    def _series_data(payload: dict, excluded_wells: set[str]) -> dict:
        """把板载荷整理为各系列的 (浓度, 读数, 稀释水平) 序列，剔除已排除孔。"""
        data: dict[str, dict] = {}
        for name in sorted(payload["series"].keys()):
            data[name] = {"x": [], "y": [], "levels": []}
        for w in payload["wells"]:  # 载荷已按 (series, level, well) 排序
            if w["well"] in excluded_wells:
                continue
            entry = data[w["series"]]
            conc = payload["series"][w["series"]]["concentrations"][w["level"]]
            entry["x"].append(conc)
            entry["y"].append(w["response"])
            entry["levels"].append(w["level"])
        return data

    @staticmethod
    def _run_analysis(payload: dict, fit_config: dict, ruleset: dict, excluded_wells: set[str]) -> dict:
        """纯函数式分析：同一 (载荷, 拟合参数, 规则集, 排除集) 必得同一结果。"""
        cfg = dict(DEFAULT_FIT_CONFIG)
        cfg.update(fit_config or {})
        rs = merged_ruleset(ruleset)
        data = PotencyService._series_data(payload, excluded_wells)
        fits = {
            name: fit_4pl(d["x"], d["y"], max_iter=cfg["max_iter"], tol=cfg["tolerance"])
            for name, d in data.items()
        }
        series_metrics: dict[str, dict] = {}
        for name, d in data.items():
            groups = group_by_level(d["levels"], d["y"])
            fit = fits[name]
            rsd = replicate_rsd(groups)
            levels = valid_levels(
                groups, fit.A, fit.D, rs["valid_range_low"], rs["valid_range_high"]
            ) if fit.converged else []
            series_metrics[name] = {
                "n_points": len(d["y"]),
                "replicate_rsd": rsd,
                "valid_levels": levels,
                "n_valid_levels": len(levels),
            }
        std_name = next(n for n in sorted(data) if payload["series"][n]["kind"] == "standard")
        samples: dict[str, dict] = {}
        for name in sorted(data):
            if payload["series"][name]["kind"] != "sample":
                continue
            par = parallelism_test(
                data[std_name]["x"], data[std_name]["y"],
                data[name]["x"], data[name]["y"],
                max_iter=cfg["max_iter"], tol=cfg["tolerance"],
            )
            pot = relative_potency(fits[std_name], fits[name])
            ctx = {
                "std_converged": fits[std_name].converged,
                "smp_converged": fits[name].converged,
                "std_status": fits[std_name].status,
                "smp_status": fits[name].status,
                "span_std": abs(fits[std_name].A - fits[std_name].D) if fits[std_name].converged else None,
                "span_smp": abs(fits[name].A - fits[name].D) if fits[name].converged else None,
                "r2_std": fits[std_name].r_squared if fits[std_name].converged else None,
                "r2_smp": fits[name].r_squared if fits[name].converged else None,
                "rsd_std": series_metrics[std_name]["replicate_rsd"]["max_rsd_pct"],
                "rsd_smp": series_metrics[name]["replicate_rsd"]["max_rsd_pct"],
                "valid_levels_std": series_metrics[std_name]["n_valid_levels"],
                "valid_levels_smp": series_metrics[name]["n_valid_levels"],
                "parallelism_p": par["p"],
                "potency_estimable": pot["estimable"],
            }
            hits = evaluate_ruleset(rs, ctx)
            samples[name] = {
                "parallelism": par,
                "potency": pot,
                "rule_hits": hits,
                "valid": all_passed(hits),
            }
        return sanitize({
            "standard_series": std_name,
            "excluded_wells": sorted(excluded_wells),
            "fits": {n: f.to_dict() for n, f in fits.items()},
            "series_metrics": series_metrics,
            "samples": samples,
        })

    def analyze_plate(self, plate_id: int, actor: str) -> dict:
        """对板执行一次新的分析版本（应用当前全部排除孔，快照当时规则集）。"""
        plate = self.get_plate(plate_id)
        batch = self.get_batch(plate["batch_id"])
        if batch["status"] == BATCH_CONCLUDED:
            raise WorkflowError("批次已出最终结论，禁止重新分析")
        method = self.get_method_version(plate["method_version_id"])
        exclusions = self.list_exclusions(plate_id)
        excluded_wells = {e["well"] for e in exclusions}
        result = self._run_analysis(plate["payload"], method["fit_config"], method["ruleset"], excluded_wells)
        version = len(self.store.all("analyses", "plate_id = ?", (plate_id,))) + 1
        analysis_id = self.store.insert("analyses", {
            "plate_id": plate_id, "version": version,
            "method_version_id": method["id"],
            "ruleset_snapshot": method["ruleset"],
            "fit_config_snapshot": method["fit_config"],
            "exclusion_ids": [e["id"] for e in exclusions],
            "result": result,
            "created_by": actor, "created_at": self._now(),
        })
        # 旧测定全部标记为被取代，新测定成为该板当前结论
        self.store.execute(
            "UPDATE determinations SET superseded = 1 WHERE analysis_id IN "
            "(SELECT id FROM analyses WHERE plate_id = ?)", (plate_id,),
        )
        for series_name, sample in result["samples"].items():
            pot = sample["potency"]
            self.store.insert("determinations", {
                "batch_id": plate["batch_id"], "analysis_id": analysis_id,
                "series": series_name,
                "log_potency": pot["log_potency"],
                "potency_pct": pot["potency_pct"],
                "se": pot["se_log_potency"],
                "df": pot["df"],
                "valid": 1 if sample["valid"] else 0,
                "superseded": 0,
                "rule_hits": sample["rule_hits"],
                "created_at": self._now(),
            })
        self._audit("analyses", analysis_id, "ANALYZE", actor, {
            "plate_id": plate_id, "version": version,
        })
        return self.store.by_id("analyses", analysis_id)

    def get_analysis(self, analysis_id: int) -> dict:
        return self._get_or_404("analyses", analysis_id, "分析")

    def list_analyses(self, plate_id: int) -> list[dict]:
        return self.store.all("analyses", "plate_id = ?", (plate_id,))

    def recompute_analysis(self, analysis_id: int) -> dict:
        """按分析记录中快照的规则集、拟合参数与排除集重算，验证结果逐位一致。"""
        analysis = self.get_analysis(analysis_id)
        plate = self.get_plate(analysis["plate_id"])
        excluded_wells = {
            e["well"] for e in (
                self.store.by_id("exclusions", eid) for eid in analysis["exclusion_ids"]
            ) if e is not None
        }
        recomputed = self._run_analysis(
            plate["payload"], analysis["fit_config_snapshot"],
            analysis["ruleset_snapshot"], excluded_wells,
        )
        matches = canonical_json(recomputed) == canonical_json(analysis["result"])
        return {
            "analysis_id": analysis_id,
            "matches": matches,
            "recomputed": recomputed,
        }

    # ------------------------------------------------------------------ 排除孔
    def exclude_well(self, plate_id: int, well: str, operator: str, reason: str) -> dict:
        """排除一个孔：登记操作者与理由，自动记录排除前后各系列效价差异。"""
        plate = self.get_plate(plate_id)
        batch = self.get_batch(plate["batch_id"])
        if batch["status"] == BATCH_CONCLUDED:
            raise WorkflowError("批次已出最终结论，禁止排除孔")
        known_wells = {w["well"] for w in plate["payload"]["wells"]}
        if well not in known_wells:
            raise NotFoundError(f"板上不存在孔位: {well}")
        if self.store.one("exclusions", "plate_id = ? AND well = ?", (plate_id, well)):
            raise WorkflowError(f"孔 {well} 已被排除，排除记录不可修改")
        if not reason or not reason.strip():
            raise ValidationError("排除孔必须填写理由")
        # 确保存在"排除前"分析
        analyses = self.list_analyses(plate_id)
        before = analyses[-1] if analyses else self.analyze_plate(plate_id, operator)
        exclusion_id = self.store.insert("exclusions", {
            "plate_id": plate_id, "well": well, "operator": operator,
            "reason": reason, "before_analysis_id": before["id"],
            "after_analysis_id": None, "before_potency": None,
            "after_potency": None, "delta_potency": None,
            "created_at": self._now(),
        })
        after = self.analyze_plate(plate_id, operator)
        before_pot = {n: s["potency"]["potency_pct"] for n, s in before["result"]["samples"].items()}
        after_pot = {n: s["potency"]["potency_pct"] for n, s in after["result"]["samples"].items()}
        delta = {
            n: (after_pot.get(n) - before_pot.get(n))
            if after_pot.get(n) is not None and before_pot.get(n) is not None else None
            for n in sorted(set(before_pot) | set(after_pot))
        }
        self.store.update("exclusions", exclusion_id, {
            "after_analysis_id": after["id"],
            "before_potency": before_pot,
            "after_potency": after_pot,
            "delta_potency": delta,
        })
        self._audit("exclusions", exclusion_id, "EXCLUDE_WELL", operator, {
            "plate_id": plate_id, "well": well, "reason": reason, "delta_potency": delta,
        })
        return self.store.by_id("exclusions", exclusion_id)

    def list_exclusions(self, plate_id: int) -> list[dict]:
        return self.store.all("exclusions", "plate_id = ?", (plate_id,))

    # ------------------------------------------------------------------ 测定与结论
    def list_determinations(self, batch_id: int, include_superseded: bool = False) -> list[dict]:
        rows = self.store.all("determinations", "batch_id = ?", (batch_id,))
        if include_superseded:
            return rows
        return [r for r in rows if not r["superseded"]]

    def _valid_determinations(self, batch_id: int) -> list[dict]:
        """全部有效且未被取代的测定 —— 结论组合的唯一合法输入。"""
        return [r for r in self.list_determinations(batch_id) if r["valid"]]

    def _combine_and_record(self, batch: dict, kind: str, actor: str) -> dict:
        method = self.get_method_version(batch["method_version_id"])
        dets = self._valid_determinations(batch["id"])
        if dets:
            combined = combine_assay(
                [{"log_potency": d["log_potency"], "se": d["se"], "df": d["df"]} for d in dets],
                method["combination_strategy"],
            )
            lo, hi = method["release_low_pct"], method["release_high_pct"]
            pct = combined["potency_pct"]
            outcome = "PASS" if lo <= pct <= hi else "FAIL"
            ci = combined["ci95_pct"] or [None, None]
        else:
            combined = None
            outcome = "NO_VALID_DETERMINATION"
            ci = [None, None]
        row_id = self.store.insert("conclusions", {
            "batch_id": batch["id"], "kind": kind,
            "strategy": method["combination_strategy"],
            "determination_ids": [d["id"] for d in dets],
            "n_determinations": len(dets),
            "combined_potency_pct": combined["potency_pct"] if combined else None,
            "ci_lower_pct": ci[0], "ci_upper_pct": ci[1],
            "outcome": outcome,
            "created_by": actor, "created_at": self._now(),
        })
        return self.store.by_id("conclusions", row_id)

    def lock_first_round(self, batch_id: int, actor: str, role: str) -> dict:
        """锁定首轮结论：此后导入新板必须走复测审批。"""
        self._require_role(role, APPROVER_ROLES, "锁定首轮结论")
        batch = self.get_batch(batch_id)
        if batch["status"] != BATCH_OPEN:
            raise WorkflowError(f"批次状态为 {batch['status']}，不能锁定首轮结论")
        if not self.list_determinations(batch_id):
            raise WorkflowError("尚无任何测定，不能锁定首轮结论")
        conclusion = self._combine_and_record(batch, "FIRST_ROUND", actor)
        self.store.update("batches", batch_id, {"status": BATCH_FIRST_ROUND_LOCKED})
        self._audit("batches", batch_id, "LOCK_FIRST_ROUND", actor, {"conclusion_id": conclusion["id"]})
        return conclusion

    def request_retest(self, batch_id: int, actor: str, role: str, reason: str) -> dict:
        """申请复测：仅首轮结论锁定后允许。"""
        batch = self.get_batch(batch_id)
        if batch["status"] != BATCH_FIRST_ROUND_LOCKED:
            raise WorkflowError("首轮结论锁定后才能申请复测")
        if not reason or not reason.strip():
            raise ValidationError("复测申请必须填写理由")
        row_id = self.store.insert("retest_requests", {
            "batch_id": batch_id, "requested_by": actor, "requester_role": role,
            "reason": reason, "status": "PENDING",
            "approved_by": None, "approver_role": None,
            "created_at": self._now(), "decided_at": None,
        })
        self._audit("retest_requests", row_id, "REQUEST", actor, {"batch_id": batch_id})
        return self.store.by_id("retest_requests", row_id)

    def decide_retest(self, request_id: int, actor: str, role: str, approve: bool) -> dict:
        """批准/拒绝复测：批准人必须与申请人不同人、不同角色。"""
        self._require_role(role, APPROVER_ROLES, "审批复测")
        req = self._get_or_404("retest_requests", request_id, "复测申请")
        if req["status"] != "PENDING":
            raise WorkflowError(f"复测申请已处理（{req['status']}），不能重复审批")
        if actor == req["requested_by"]:
            raise WorkflowError("批准人不能是申请人本人")
        if role == req["requester_role"]:
            raise WorkflowError("批准角色必须与申请角色不同")
        self.store.update("retest_requests", request_id, {
            "status": "APPROVED" if approve else "REJECTED",
            "approved_by": actor, "approver_role": role,
            "decided_at": self._now(),
        })
        self._audit("retest_requests", request_id, "APPROVE" if approve else "REJECT", actor, {})
        return self.store.by_id("retest_requests", request_id)

    def get_retest_request(self, request_id: int) -> dict:
        return self._get_or_404("retest_requests", request_id, "复测申请")

    def conclude_batch(self, batch_id: int, actor: str, role: str) -> dict:
        """形成最终结论：纳入当前全部有效测定（首轮 + 复测），无挑选入口。"""
        self._require_role(role, APPROVER_ROLES, "形成最终结论")
        batch = self.get_batch(batch_id)
        if batch["status"] == BATCH_CONCLUDED:
            raise WorkflowError("批次已有最终结论")
        if batch["status"] != BATCH_FIRST_ROUND_LOCKED:
            raise WorkflowError("请先锁定首轮结论，再形成最终结论")
        if not self._valid_determinations(batch_id):
            raise WorkflowError("没有有效测定，无法形成最终结论")
        conclusion = self._combine_and_record(batch, "FINAL", actor)
        self.store.update("batches", batch_id, {"status": BATCH_CONCLUDED})
        self._audit("batches", batch_id, "CONCLUDE", actor, {"conclusion_id": conclusion["id"]})
        return conclusion

    def list_conclusions(self, batch_id: int) -> list[dict]:
        return self.store.all("conclusions", "batch_id = ?", (batch_id,))

    # ------------------------------------------------------------------ 追溯
    def trace_batch(self, batch_id: int) -> dict:
        """从批次结论回溯到原始读数，并逐板核验内容哈希（证明读数未被修改）。"""
        batch = self.get_batch(batch_id)
        method = self.get_method_version(batch["method_version_id"])
        plates = []
        for plate in self.list_plates(batch_id):
            recomputed = plate_content_hash(plate["payload"])
            analyses = self.list_analyses(plate["id"])
            plates.append({
                "id": plate["id"],
                "plate_label": plate["plate_label"],
                "content_hash": plate["content_hash"],
                "hash_verified": recomputed == plate["content_hash"],
                "standard_lot": plate["payload"]["standard_lot"],
                "imported_by": plate["imported_by"],
                "imported_at": plate["imported_at"],
                "retest_request_id": plate["retest_request_id"],
                "exclusions": self.list_exclusions(plate["id"]),
                "analyses": [
                    {
                        "id": a["id"], "version": a["version"],
                        "ruleset_snapshot": a["ruleset_snapshot"],
                        "fit_config_snapshot": a["fit_config_snapshot"],
                        "exclusion_ids": a["exclusion_ids"],
                        "created_by": a["created_by"], "created_at": a["created_at"],
                    }
                    for a in analyses
                ],
            })
        determinations = self.list_determinations(batch_id, include_superseded=True)
        conclusions = self.list_conclusions(batch_id)
        return {
            "batch": batch,
            "method_version": method,
            "conclusions": conclusions,
            "determinations": determinations,
            "plates": plates,
            "integrity": {
                "all_raw_readings_intact": all(p["hash_verified"] for p in plates),
                "n_plates": len(plates),
            },
        }
