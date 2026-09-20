"""会审业务逻辑。

工作流与防选择性复测设计：

1. ``import_plate`` 保存方法版本、板图、标准品批号、稀释序列、原始读数；
   规范化内容哈希用于**重复导入识别**，原始读数受数据库触发器保护。
2. ``create_analysis`` 产生**只追加**的分析版本：四参数拟合 + 当时生效
   规则集评估；每个排除孔记录操作者、理由及该孔排除前后的效价差异。
3. ``lock_first_round`` 锁定首轮结论（合格/无效），锁定后该板冻结。
4. 仅锁定板可提 ``create_retest_request``；复测须由**不同于申请人**的
   SUPERVISOR 与 QA 两个角色分别批准。
5. 复测板继承谱系（lineage），批次结论按预置组合策略纳入谱系内**全部**
   有效测定——无效板不能被剔除，悬挂的复测申请必须先了结，禁止挑数据。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime
from typing import Callable, Optional

from . import db as dbmod
from .fitting import fit_relative_potency
from .rules import (
    DEFAULT_CONTEXT,
    DEFAULT_RULES,
    RuleSet,
    evaluate_plate,
    ruleset_effective_on,
)

WELL_RE = re.compile(r"^[A-Z]{1,2}[0-9]{1,2}$")
ROLES = ("standard", "sample")


class ServiceError(Exception):
    """业务规则冲突（4xx 语义）。"""

    def __init__(self, message: str, code: str = "SERVICE_ERROR", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class DuplicateImport(ServiceError):
    def __init__(self, existing_plate_id: int, plate_code: str, content_hash: str):
        super().__init__(
            f"重复导入：内容哈希 {content_hash[:12]}… 已存在于板 {plate_code}",
            code="DUPLICATE_IMPORT",
            status=409,
        )
        self.existing_plate_id = existing_plate_id
        self.existing_plate_code = plate_code
        self.content_hash = content_hash


def _canonical_hash(payload: dict) -> str:
    """对板关键元数据 + 全部读数做规范化 SHA-256（键排序、分隔确定）。"""
    wells = sorted(payload["wells"], key=lambda w: w["well"])
    parts = [
        payload["method_code"],
        payload["method_version"],
        payload["standard_lot"],
        payload["sample_lot"],
        f"assigned={float(payload['assigned_potency'])!r}",
    ]
    for w in wells:
        parts.append(
            f"{w['well']}|{w['role']}|{float(w['dose'])!r}|{float(w['reading'])!r}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class Service:
    def __init__(self, conn, clock: Optional[Callable[[], datetime]] = None):
        self.conn = conn
        self.clock = clock or (lambda: datetime.now())

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return self.clock().isoformat()

    def _today(self) -> date:
        return self.clock().date()

    def _audit(self, entity, entity_id, action, actor, detail=None):
        self.conn.execute(
            """INSERT INTO audit_log(entity, entity_id, action, actor, at, detail_json)
               VALUES (?,?,?,?,?,?)""",
            (
                entity,
                entity_id,
                action,
                actor,
                self._now(),
                json.dumps(detail or {}, ensure_ascii=False, default=str),
            ),
        )

    def _user(self, username: str, require_role: Optional[str] = None):
        row = self.conn.execute(
            "SELECT username, role FROM users WHERE username=?", (username,)
        ).fetchone()
        if row is None:
            raise ServiceError(f"用户 {username} 不存在", "UNKNOWN_USER", 401)
        if require_role and row["role"] != require_role:
            raise ServiceError(
                f"需要 {require_role} 角色，{username} 为 {row['role']}",
                "FORBIDDEN_ROLE",
                403,
            )
        return row["role"]

    def _ruleset_row(self, on_date: date):
        rows = self.conn.execute(
            """SELECT * FROM rule_sets WHERE code='CELL_POTENCY'
               AND effective_from <= ? ORDER BY effective_from DESC, version DESC""",
            (on_date.isoformat(),),
        ).fetchall()
        if not rows:
            raise ServiceError("当日无生效规则集", "NO_RULESET", 500)
        return rows[0]

    def _load_ruleset(self, rs_row) -> RuleSet:
        # 以代码内规则定义为准（版本即代码版本），保证复算一致
        match = [
            r
            for r in DEFAULT_RULES
            if r.code == rs_row["code"] and r.version == rs_row["version"]
        ]
        if not match:
            raise ServiceError(
                f"规则集 {rs_row['code']} {rs_row['version']} 未在代码中定义",
                "RULESET_UNAVAILABLE",
                500,
            )
        return match[0]

    # ------------------------------------------------------------------
    # 1. 导板（含重复识别）
    # ------------------------------------------------------------------

    def import_plate(self, payload: dict, actor: str) -> dict:
        self._user(actor, "ANALYST")
        self._validate_import_payload(payload)

        method = self.conn.execute(
            "SELECT id FROM methods WHERE code=? AND version=?",
            (payload["method_code"], payload["method_version"]),
        ).fetchone()
        if method is None:
            raise ServiceError(
                f"方法版本 {payload['method_code']} {payload['method_version']} 不存在",
                "UNKNOWN_METHOD",
                404,
            )

        content_hash = _canonical_hash(payload)
        dup = self.conn.execute(
            "SELECT id, plate_code FROM plates WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if dup is not None:
            raise DuplicateImport(dup["id"], dup["plate_code"], content_hash)

        retest_request_id = payload.get("retest_request_id")
        lineage_key = uuid.uuid4().hex
        if retest_request_id is not None:
            req = self.conn.execute(
                "SELECT * FROM retest_requests WHERE id=?", (retest_request_id,)
            ).fetchone()
            if req is None or req["status"] != "approved":
                raise ServiceError(
                    "复测板必须关联已批准的复测申请", "REQUEST_NOT_APPROVED", 409
                )
            lineage_key = req["lineage_key"]

        plate_code = payload["plate_code"]
        if self.conn.execute(
            "SELECT 1 FROM plates WHERE plate_code=?", (plate_code,)
        ).fetchone():
            raise ServiceError(f"板号 {plate_code} 已存在", "PLATE_CODE_TAKEN", 409)

        wells = sorted(payload["wells"], key=lambda w: w["well"])
        layout = {
            w["well"]: {"role": w["role"], "dose": float(w["dose"])} for w in wells
        }
        dilution = {
            role: sorted({float(w["dose"]) for w in wells if w["role"] == role})
            for role in ROLES
        }
        ts = self._now()
        cur = self.conn.execute(
            """INSERT INTO plates(plate_code, method_id, standard_lot, sample_lot,
               assigned_potency, dilution_json, layout_json, status, content_hash,
               imported_by, imported_at, retest_of_request_id, lineage_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                plate_code,
                method["id"],
                payload["standard_lot"],
                payload["sample_lot"],
                float(payload["assigned_potency"]),
                json.dumps(dilution, ensure_ascii=False),
                json.dumps(layout, ensure_ascii=False),
                "imported",
                content_hash,
                actor,
                ts,
                retest_request_id,
                lineage_key,
            ),
        )
        plate_id = cur.lastrowid
        self.conn.executemany(
            """INSERT INTO plate_readings(plate_id, well, role, dose, reading)
               VALUES (?,?,?,?,?)""",
            [
                (
                    plate_id,
                    w["well"],
                    w["role"],
                    float(w["dose"]),
                    float(w["reading"]),
                )
                for w in wells
            ],
        )
        self.conn.execute(
            """INSERT INTO raw_imports(plate_id, source_name, content_hash,
               payload_json, imported_by, imported_at)
               VALUES (?,?,?,?,?,?)""",
            (
                plate_id,
                payload.get("source_name", plate_code),
                content_hash,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                actor,
                ts,
            ),
        )
        if retest_request_id is not None:
            self.conn.execute(
                "INSERT INTO retest_plates(request_id, plate_id) VALUES (?,?)",
                (retest_request_id, plate_id),
            )
        self._audit("plate", plate_id, "import", actor, {"plate_code": plate_code})
        self.conn.commit()
        return self.get_plate(plate_id)

    def _validate_import_payload(self, p: dict):
        required = [
            "plate_code",
            "method_code",
            "method_version",
            "standard_lot",
            "sample_lot",
            "assigned_potency",
            "wells",
        ]
        for k in required:
            if k not in p:
                raise ServiceError(f"导入载荷缺少字段 {k}", "BAD_PAYLOAD")
        if float(p["assigned_potency"]) <= 0:
            raise ServiceError("标示效价必须为正数", "BAD_PAYLOAD")
        wells = p["wells"]
        if not isinstance(wells, list) or not wells:
            raise ServiceError("wells 必须为非空列表", "BAD_PAYLOAD")
        seen = set()
        for w in wells:
            for k in ("well", "role", "dose", "reading"):
                if k not in w:
                    raise ServiceError(f"孔记录缺少 {k}", "BAD_PAYLOAD")
            if not WELL_RE.match(w["well"]):
                raise ServiceError(f"孔位 {w['well']} 格式非法", "BAD_PAYLOAD")
            if w["role"] not in ROLES:
                raise ServiceError(f"孔 {w['well']} 角色非法", "BAD_PAYLOAD")
            if float(w["dose"]) <= 0:
                raise ServiceError(f"孔 {w['well']} 剂量必须为正", "BAD_PAYLOAD")
            if w["well"] in seen:
                raise ServiceError(f"孔位 {w['well']} 重复", "BAD_PAYLOAD")
            seen.add(w["well"])

    # ------------------------------------------------------------------
    # 2. 分析版本（拟合 + 规则 + 排孔审计）
    # ------------------------------------------------------------------

    def _active_wells(self, plate_id: int, excluded: set[str]):
        rows = self.conn.execute(
            "SELECT well, role, dose, reading FROM plate_readings WHERE plate_id=?",
            (plate_id,),
        ).fetchall()
        return [r for r in rows if r["well"] not in excluded]

    def create_analysis(
        self,
        plate_id: int,
        actor: str,
        exclusions: Optional[list[dict]] = None,
        note: str = "",
    ) -> dict:
        self._user(actor, "ANALYST")
        plate = self.conn.execute(
            "SELECT * FROM plates WHERE id=?", (plate_id,)
        ).fetchone()
        if plate is None:
            raise ServiceError("板不存在", "NOT_FOUND", 404)
        if plate["status"] in ("locked", "invalid_locked"):
            raise ServiceError("板已锁定，不能新增分析版本", "PLATE_LOCKED", 409)

        exclusions = exclusions or []
        # 规范化 + 校验排孔申请
        well_rows = {
            r["well"]: r
            for r in self.conn.execute(
                "SELECT well, role, dose, reading FROM plate_readings WHERE plate_id=?",
                (plate_id,),
            ).fetchall()
        }
        excl_seen = set()
        for e in exclusions:
            for k in ("well", "operator", "reason"):
                if not e.get(k):
                    raise ServiceError(f"排孔记录缺少 {k}", "BAD_PAYLOAD")
            if e["well"] not in well_rows:
                raise ServiceError(
                    f"排孔 {e['well']} 不在板上", "UNKNOWN_WELL", 404
                )
            if e["well"] in excl_seen:
                raise ServiceError(f"排孔 {e['well']} 重复", "BAD_PAYLOAD")
            excl_seen.add(e["well"])

        rs_row = self._ruleset_row(self._today())
        ruleset = self._load_ruleset(rs_row)
        assigned = plate["assigned_potency"]

        # 按孔位确定顺序逐个排孔，记录每孔的前后效价差异
        audit_rows = []
        current_excluded: set[str] = set()
        ordered = sorted(exclusions, key=lambda e: e["well"])
        for e in ordered:
            before = self._safe_fit(
                self._active_wells(plate_id, current_excluded), assigned
            )
            current_excluded.add(e["well"])
            after = self._safe_fit(
                self._active_wells(plate_id, current_excluded), assigned
            )
            p_before = before.get("relative_potency")
            p_after = after.get("relative_potency")
            delta = (
                (p_after - p_before) / p_before * 100.0
                if p_before and p_after
                else None
            )
            wr = well_rows[e["well"]]
            audit_rows.append(
                {
                    "well": e["well"],
                    "role": wr["role"],
                    "dose": wr["dose"],
                    "reading": wr["reading"],
                    "operator": e["operator"],
                    "reason": e["reason"],
                    "potency_before": p_before,
                    "potency_after": p_after,
                    "delta_potency_pct": delta,
                }
            )

        fit = self._safe_fit(
            self._active_wells(plate_id, current_excluded), assigned
        )
        evaluation = evaluate_plate(fit, ruleset)

        version_no = self.conn.execute(
            "SELECT COALESCE(MAX(version_no),0)+1 AS v FROM analyses WHERE plate_id=?",
            (plate_id,),
        ).fetchone()["v"]
        ts = self._now()
        cur = self.conn.execute(
            """INSERT INTO analyses(plate_id, version_no, ruleset_id, fit_json,
               eval_json, note, created_by, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                plate_id,
                version_no,
                rs_row["id"],
                json.dumps(fit, ensure_ascii=False, default=_float_default),
                json.dumps(evaluation.to_dict(), ensure_ascii=False),
                note,
                actor,
                ts,
            ),
        )
        analysis_id = cur.lastrowid
        self.conn.executemany(
            """INSERT INTO analysis_exclusions(analysis_id, well, role, dose, reading,
               operator, reason, potency_before, potency_after,
               delta_potency_pct, excluded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    analysis_id,
                    a["well"],
                    a["role"],
                    a["dose"],
                    a["reading"],
                    a["operator"],
                    a["reason"],
                    a["potency_before"],
                    a["potency_after"],
                    a["delta_potency_pct"],
                    ts,
                )
                for a in audit_rows
            ],
        )
        self.conn.execute(
            "UPDATE plates SET status='analyzed', analyzed_at=? WHERE id=?",
            (ts, plate_id),
        )
        self._audit(
            "analysis",
            analysis_id,
            "create",
            actor,
            {"plate_id": plate_id, "version_no": version_no,
             "excluded": sorted(excl_seen)},
        )
        self.conn.commit()
        return self.get_analysis(analysis_id)

    def _safe_fit(self, well_rows, assigned: float) -> dict:
        xs, ys, xm, ym = [], [], [], []
        for r in well_rows:
            if r["role"] == "standard":
                xs.append(r["dose"])
                ys.append(r["reading"])
            else:
                xm.append(r["dose"])
                ym.append(r["reading"])
        try:
            result = fit_relative_potency(xs, ys, xm, ym, float(assigned))
            return result.to_dict()
        except ValueError as exc:
            return _failed_fit(float(assigned), str(exc))

    # ------------------------------------------------------------------
    # 3. 锁定首轮结论
    # ------------------------------------------------------------------

    def lock_first_round(self, plate_id: int, actor: str,
                         analysis_version: Optional[int] = None) -> dict:
        self._user(actor, "SUPERVISOR")
        plate = self._plate_or_404(plate_id)
        if plate["status"] in ("locked", "invalid_locked"):
            raise ServiceError("板已锁定", "PLATE_LOCKED", 409)
        if plate["status"] != "analyzed":
            raise ServiceError("板尚无分析，不能锁定", "NOT_ANALYZED", 409)
        if analysis_version is None:
            analysis_version = self.conn.execute(
                "SELECT MAX(version_no) AS v FROM analyses WHERE plate_id=?",
                (plate_id,),
            ).fetchone()["v"]
        analysis = self.conn.execute(
            "SELECT * FROM analyses WHERE plate_id=? AND version_no=?",
            (plate_id, analysis_version),
        ).fetchone()
        if analysis is None:
            raise ServiceError("分析版本不存在", "NOT_FOUND", 404)
        ev = json.loads(analysis["eval_json"])
        new_status = "locked" if ev["valid"] else "invalid_locked"
        ts = self._now()
        self.conn.execute(
            "UPDATE plates SET status=?, locked_at=?, locked_analysis_id=? WHERE id=?",
            (new_status, ts, analysis["id"], plate_id),
        )
        self._audit(
            "plate",
            plate_id,
            "lock_first_round",
            actor,
            {"analysis_id": analysis["id"], "conclusion": new_status},
        )
        self.conn.commit()
        return self.get_plate(plate_id)

    # ------------------------------------------------------------------
    # 4. 复测申请与多角色审批
    # ------------------------------------------------------------------

    def create_retest_request(self, plate_id: int, actor: str, reason: str) -> dict:
        # 申请人可为分析员或主管；审批仅 SUPERVISOR/QA，且申请人不得自批
        role = self._user(actor)
        if role not in ("ANALYST", "SUPERVISOR"):
            raise ServiceError(
                "仅 ANALYST 或 SUPERVISOR 可发起复测申请", "FORBIDDEN_ROLE", 403
            )
        plate = self._plate_or_404(plate_id)
        if plate["status"] not in ("locked", "invalid_locked"):
            raise ServiceError(
                "只有首轮结论锁定后的板才能申请复测", "PLATE_NOT_LOCKED", 409
            )
        if not reason or not reason.strip():
            raise ServiceError("复测理由必填", "BAD_PAYLOAD")
        open_req = self.conn.execute(
            """SELECT id FROM retest_requests WHERE plate_id=?
               AND status='pending'""",
            (plate_id,),
        ).fetchone()
        if open_req:
            raise ServiceError("该板已有待审批的复测申请", "REQUEST_EXISTS", 409)
        ts = self._now()
        cur = self.conn.execute(
            """INSERT INTO retest_requests(plate_id, lineage_key, requested_by,
               reason, status, created_at) VALUES (?,?,?,?, 'pending', ?)""",
            (plate_id, plate["lineage_key"], actor, reason.strip(), ts),
        )
        req_id = cur.lastrowid
        self._audit("retest_request", req_id, "create", actor,
                    {"plate_id": plate_id, "reason": reason})
        self.conn.commit()
        return self.get_retest_request(req_id)

    def decide_retest(self, request_id: int, actor: str, decision: str,
                      comment: str = "") -> dict:
        role = self._user(actor)
        if role not in ("SUPERVISOR", "QA"):
            raise ServiceError("仅 SUPERVISOR 或 QA 可审批复测", "FORBIDDEN_ROLE", 403)
        if decision not in ("approved", "rejected"):
            raise ServiceError("decision 必须为 approved/rejected", "BAD_PAYLOAD")
        req = self.conn.execute(
            "SELECT * FROM retest_requests WHERE id=?", (request_id,)
        ).fetchone()
        if req is None:
            raise ServiceError("复测申请不存在", "NOT_FOUND", 404)
        if req["status"] != "pending":
            raise ServiceError("申请已了结", "REQUEST_CLOSED", 409)
        if actor == req["requested_by"]:
            raise ServiceError(
                "申请人不能审批自己的复测申请（职责分离）", "SELF_APPROVAL", 403
            )
        exists = self.conn.execute(
            "SELECT 1 FROM retest_approvals WHERE request_id=? AND approver_role=?",
            (request_id, role),
        ).fetchone()
        if exists:
            raise ServiceError(f"{role} 已审批过该申请", "ALREADY_APPROVED", 409)

        ts = self._now()
        self.conn.execute(
            """INSERT INTO retest_approvals(request_id, approver_user, approver_role,
               decision, comment, decided_at) VALUES (?,?,?,?,?,?)""",
            (request_id, actor, role, decision, comment, ts),
        )

        if decision == "rejected":
            self.conn.execute(
                "UPDATE retest_requests SET status='rejected', decided_at=? WHERE id=?",
                (ts, request_id),
            )
            self._audit("retest_request", request_id, "reject", actor, {"role": role})
            self.conn.commit()
            return self.get_retest_request(request_id)

        approvals = self.conn.execute(
            "SELECT approver_role FROM retest_approvals WHERE request_id=? AND decision='approved'",
            (request_id,),
        ).fetchall()
        roles_ok = {r["approver_role"] for r in approvals}
        if {"SUPERVISOR", "QA"}.issubset(roles_ok):
            self.conn.execute(
                "UPDATE retest_requests SET status='approved', decided_at=? WHERE id=?",
                (ts, request_id),
            )
            self._audit("retest_request", request_id, "approve", actor,
                        {"roles": sorted(roles_ok)})
        else:
            self._audit("retest_request", request_id, "partial_approve", actor,
                        {"role": role, "waiting_for": sorted(
                            {"SUPERVISOR", "QA"} - roles_ok)})
        self.conn.commit()
        return self.get_retest_request(request_id)

    # ------------------------------------------------------------------
    # 5. 批次与组合结论
    # ------------------------------------------------------------------

    def create_batch(self, batch_code: str, product: str, strategy_code: str,
                     actor: str) -> dict:
        self._user(actor, "ANALYST")
        strat = self.conn.execute(
            "SELECT id FROM combination_strategies WHERE code=?", (strategy_code,)
        ).fetchone()
        if strat is None:
            raise ServiceError(f"策略 {strategy_code} 不存在", "UNKNOWN_STRATEGY", 404)
        if self.conn.execute("SELECT 1 FROM batches WHERE batch_code=?",
                             (batch_code,)).fetchone():
            raise ServiceError("批次号已存在", "BATCH_EXISTS", 409)
        ts = self._now()
        cur = self.conn.execute(
            """INSERT INTO batches(batch_code, product, strategy_id, status,
               created_by, created_at) VALUES (?,?,?, 'open', ?,?)""",
            (batch_code, product, strat["id"], actor, ts),
        )
        batch_id = cur.lastrowid
        self._audit("batch", batch_id, "create", actor, {"strategy": strategy_code})
        self.conn.commit()
        return self.get_batch(batch_id)

    def add_plate_to_batch(self, batch_id: int, plate_id: int, actor: str) -> dict:
        self._user(actor, "ANALYST")
        batch = self._batch_or_404(batch_id)
        if batch["status"] != "open":
            raise ServiceError("批次已结论", "BATCH_CLOSED", 409)
        plate = self._plate_or_404(plate_id)
        if self.conn.execute(
            "SELECT 1 FROM batch_plates WHERE batch_id=? AND plate_id=?",
            (batch_id, plate_id),
        ).fetchone():
            return self.get_batch(batch_id)
        ts = self._now()
        self.conn.execute(
            "INSERT INTO batch_plates(batch_id, plate_id, added_at) VALUES (?,?,?)",
            (batch_id, plate_id, ts),
        )
        self._audit("batch", batch_id, "add_plate", actor, {"plate_id": plate_id})
        self.conn.commit()
        return self.get_batch(batch_id)

    def conclude_batch(self, batch_id: int, actor: str) -> dict:
        self._user(actor, "QA")
        batch = self._batch_or_404(batch_id)
        if batch["status"] != "open":
            raise ServiceError("批次已结论", "BATCH_CLOSED", 409)

        plate_rows = self.conn.execute(
            """SELECT p.* FROM batch_plates bp JOIN plates p ON p.id=bp.plate_id
               WHERE bp.batch_id=? ORDER BY p.id""",
            (batch_id,),
        ).fetchall()
        if not plate_rows:
            raise ServiceError("批次内没有测定板", "NO_PLATES", 409)

        included_ids = [p["id"] for p in plate_rows]
        lineages = {p["lineage_key"] for p in plate_rows}

        # 反挑选核查 1：谱系内所有板都必须纳入（不能只留有利板）
        for lk in lineages:
            siblings = self.conn.execute(
                "SELECT id, plate_code, status FROM plates WHERE lineage_key=? ORDER BY id",
                (lk,),
            ).fetchall()
            for s in siblings:
                if s["id"] not in included_ids:
                    raise ServiceError(
                        f"谱系 {lk[:8]} 中的板 {s['plate_code']} 未纳入批次，"
                        "禁止只挑选有利结果",
                        "LINEAGE_INCOMPLETE",
                        409,
                    )
                if s["status"] not in ("locked", "invalid_locked"):
                    raise ServiceError(
                        f"板 {s['plate_code']} 尚未锁定首轮结论", "PLATE_UNLOCKED", 409
                    )

        # 反挑选核查 2：谱系不能存在悬挂的复测申请
        pending = self.conn.execute(
            """SELECT r.id FROM retest_requests r WHERE r.lineage_key IN (%s)
               AND r.status='pending'"""
            % ",".join("?" * len(lineages)),
            tuple(lineages),
        ).fetchall()
        if pending:
            raise ServiceError(
                f"存在 {len(pending)} 个待审批复测申请，须先了结", "PENDING_RETEST", 409
            )
        # 注：已批准复测产生的板与首轮板共享 lineage_key，核查 1
        # （LINEAGE_INCOMPLETE）已强制其全部纳入，无需重复检查。

        strategy = json.loads(
            self.conn.execute(
                "SELECT config_json FROM combination_strategies WHERE id=?",
                (batch["strategy_id"],),
            ).fetchone()["config_json"]
        )

        contributions = []
        pcts = []
        all_valid = True
        for p in plate_rows:
            ana = self.conn.execute(
                "SELECT * FROM analyses WHERE id=?", (p["locked_analysis_id"],)
            ).fetchone()
            ev = json.loads(ana["eval_json"])
            fit = json.loads(ana["fit_json"])
            contributions.append(
                {
                    "plate_id": p["id"],
                    "plate_code": p["plate_code"],
                    "lineage_key": p["lineage_key"],
                    "valid": ev["valid"],
                    "release_passed": ev["release_passed"],
                    "potency_percent": ev["potency_percent"],
                    "relative_potency": fit.get("relative_potency"),
                    "ci95": fit.get("ci95"),
                    "ruleset_version": ev["ruleset_version"],
                    "rule_hits": ev["hits"],
                    "retest_of_request_id": p["retest_of_request_id"],
                }
            )
            all_valid = all_valid and ev["valid"]
            if ev["potency_percent"] is not None:
                pcts.append(ev["potency_percent"])

        if strategy.get("aggregation") == "geometric_mean":
            import math

            combined_pct = math.exp(sum(math.log(v) for v in pcts) / len(pcts))
            agg_name = "geometric_mean"
        else:
            combined_pct = sum(pcts) / len(pcts)
            agg_name = "arithmetic_mean"

        lo, hi = strategy["release_range_pct"]
        released = (
            all_valid
            and strategy.get("all_plates_must_be_valid", True)
            and lo <= combined_pct <= hi
        )

        assigned_values = {p["assigned_potency"] for p in plate_rows}
        conclusion = {
            "aggregation": agg_name,
            "n_plates": len(plate_rows),
            "n_lineages": len(lineages),
            "combined_potency_percent": combined_pct,
            "release_range_pct": [lo, hi],
            "all_plates_valid": all_valid,
            "released": released,
            "assigned_potency_values": sorted(assigned_values),
            "contributions": contributions,
            "rule": "纳入全部有效（及全部无效）锁定板；无效板不可剔除",
        }
        ts = self._now()
        self.conn.execute(
            """UPDATE batches SET status='concluded', conclusion_json=?,
               concluded_at=? WHERE id=?""",
            (json.dumps(conclusion, ensure_ascii=False), ts, batch_id),
        )
        self._audit("batch", batch_id, "conclude", actor,
                    {"released": released, "combined_pct": combined_pct})
        self.conn.commit()
        return self.get_batch(batch_id)

    # ------------------------------------------------------------------
    # 查询 / 复算 / 比较
    # ------------------------------------------------------------------

    def _plate_or_404(self, plate_id: int):
        row = self.conn.execute(
            "SELECT * FROM plates WHERE id=?", (plate_id,)
        ).fetchone()
        if row is None:
            raise ServiceError("板不存在", "NOT_FOUND", 404)
        return row

    def _batch_or_404(self, batch_id: int):
        row = self.conn.execute(
            "SELECT * FROM batches WHERE id=?", (batch_id,)
        ).fetchone()
        if row is None:
            raise ServiceError("批次不存在", "NOT_FOUND", 404)
        return row

    def get_plate(self, plate_id: int) -> dict:
        p = self._plate_or_404(plate_id)
        method = self.conn.execute(
            "SELECT code, version FROM methods WHERE id=?", (p["method_id"],)
        ).fetchone()
        readings = [
            dict(r)
            for r in self.conn.execute(
                "SELECT well, role, dose, reading FROM plate_readings WHERE plate_id=? ORDER BY well",
                (plate_id,),
            ).fetchall()
        ]
        raws = [
            {
                "id": r["id"],
                "source_name": r["source_name"],
                "content_hash": r["content_hash"],
                "imported_by": r["imported_by"],
                "imported_at": r["imported_at"],
            }
            for r in self.conn.execute(
                "SELECT * FROM raw_imports WHERE plate_id=? ORDER BY id", (plate_id,)
            ).fetchall()
        ]
        out = {
            "id": p["id"],
            "plate_code": p["plate_code"],
            "method": {"code": method["code"], "version": method["version"]},
            "standard_lot": p["standard_lot"],
            "sample_lot": p["sample_lot"],
            "assigned_potency": p["assigned_potency"],
            "dilution": json.loads(p["dilution_json"]),
            "layout": json.loads(p["layout_json"]),
            "status": p["status"],
            "content_hash": p["content_hash"],
            "lineage_key": p["lineage_key"],
            "retest_of_request_id": p["retest_of_request_id"],
            "imported_by": p["imported_by"],
            "imported_at": p["imported_at"],
            "analyzed_at": p["analyzed_at"],
            "locked_at": p["locked_at"],
            "locked_analysis_id": p["locked_analysis_id"],
            "readings": readings,
            "raw_imports": raws,
        }
        return out

    def get_analysis(self, analysis_id: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM analyses WHERE id=?", (analysis_id,)
        ).fetchone()
        if row is None:
            raise ServiceError("分析不存在", "NOT_FOUND", 404)
        rs = self.conn.execute(
            "SELECT code, version FROM rule_sets WHERE id=?", (row["ruleset_id"],)
        ).fetchone()
        exclusions = [
            dict(r)
            for r in self.conn.execute(
                """SELECT well, role, dose, reading, operator, reason,
                          potency_before, potency_after, delta_potency_pct,
                          excluded_at
                   FROM analysis_exclusions WHERE analysis_id=? ORDER BY well""",
                (analysis_id,),
            ).fetchall()
        ]
        return {
            "id": row["id"],
            "plate_id": row["plate_id"],
            "version_no": row["version_no"],
            "ruleset": {"code": rs["code"], "version": rs["version"]},
            "fit": json.loads(row["fit_json"]),
            "evaluation": json.loads(row["eval_json"]),
            "note": row["note"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "exclusions": exclusions,
        }

    def list_plate_analyses(self, plate_id: int) -> list:
        self._plate_or_404(plate_id)
        rows = self.conn.execute(
            "SELECT id FROM analyses WHERE plate_id=? ORDER BY version_no", (plate_id,)
        ).fetchall()
        return [self.get_analysis(r["id"]) for r in rows]

    def compare_analyses(self, analysis_id_a: int, analysis_id_b: int) -> dict:
        """比较两个分析版本：曲线参数、效价、规则命中、排孔集合差异。"""
        a, b = self.get_analysis(analysis_id_a), self.get_analysis(analysis_id_b)

        def curve_summary(fit):
            return {
                role: {
                    "A": fit[role]["A"],
                    "B": fit[role]["B"],
                    "C_ec50": fit[role]["ec50"],
                    "D": fit[role]["D"],
                    "pooled_cv_percent": fit[role]["pooled_cv_percent"],
                    "max_cv_percent": fit[role]["max_cv_percent"],
                }
                for role in ("standard", "sample")
            }

        def hit_map(ev):
            return {h["code"]: h for h in ev["hits"]}

        ha, hb = hit_map(a["evaluation"]), hit_map(b["evaluation"])
        ea = {e["well"] for e in a["exclusions"]}
        eb = {e["well"] for e in b["exclusions"]}
        return {
            "a": {"analysis_id": a["id"], "version_no": a["version_no"]},
            "b": {"analysis_id": b["id"], "version_no": b["version_no"]},
            "curves": {
                "a": curve_summary(a["fit"]),
                "b": curve_summary(b["fit"]),
            },
            "potency": {
                "a": a["fit"].get("relative_potency"),
                "b": b["fit"].get("relative_potency"),
                "a_percent": a["evaluation"]["potency_percent"],
                "b_percent": b["evaluation"]["potency_percent"],
            },
            "parallelism": {
                "a": a["fit"].get("parallelism"),
                "b": b["fit"].get("parallelism"),
            },
            "rule_hits_added_in_b": [hb[k] for k in sorted(hb.keys() - ha.keys())],
            "rule_hits_removed_in_b": [ha[k] for k in sorted(ha.keys() - hb.keys())],
            "rule_hits_common": sorted(ha.keys() & hb.keys()),
            "exclusions_added_in_b": sorted(eb - ea),
            "exclusions_removed_in_b": sorted(ea - eb),
            "validity": {
                "a_valid": a["evaluation"]["valid"],
                "b_valid": b["evaluation"]["valid"],
            },
        }

    def get_retest_request(self, request_id: int) -> dict:
        r = self.conn.execute(
            "SELECT * FROM retest_requests WHERE id=?", (request_id,)
        ).fetchone()
        if r is None:
            raise ServiceError("复测申请不存在", "NOT_FOUND", 404)
        approvals = [
            dict(x)
            for x in self.conn.execute(
                """SELECT approver_user, approver_role, decision, comment, decided_at
                   FROM retest_approvals WHERE request_id=? ORDER BY decided_at""",
                (request_id,),
            ).fetchall()
        ]
        plates = [
            x["plate_id"]
            for x in self.conn.execute(
                "SELECT plate_id FROM retest_plates WHERE request_id=?", (request_id,)
            ).fetchall()
        ]
        return {
            "id": r["id"],
            "plate_id": r["plate_id"],
            "lineage_key": r["lineage_key"],
            "requested_by": r["requested_by"],
            "reason": r["reason"],
            "status": r["status"],
            "created_at": r["created_at"],
            "decided_at": r["decided_at"],
            "approvals": approvals,
            "retest_plate_ids": plates,
            "required_approver_roles": ["SUPERVISOR", "QA"],
        }

    def get_batch(self, batch_id: int) -> dict:
        b = self._batch_or_404(batch_id)
        plates = [
            self.get_plate(r["plate_id"])
            for r in self.conn.execute(
                "SELECT plate_id FROM batch_plates WHERE batch_id=? ORDER BY plate_id",
                (batch_id,),
            ).fetchall()
        ]
        return {
            "id": b["id"],
            "batch_code": b["batch_code"],
            "product": b["product"],
            "status": b["status"],
            "created_by": b["created_by"],
            "created_at": b["created_at"],
            "concluded_at": b["concluded_at"],
            "plates": plates,
            "conclusion": json.loads(b["conclusion_json"]) if b["conclusion_json"] else None,
        }

    def list_audit(self, entity: Optional[str] = None,
                   entity_id: Optional[int] = None) -> list:
        q = "SELECT * FROM audit_log WHERE 1=1"
        args = []
        if entity:
            q += " AND entity=?"
            args.append(entity)
        if entity_id is not None:
            q += " AND entity_id=?"
            args.append(entity_id)
        q += " ORDER BY id"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def trace_batch_readings(self, batch_id: int) -> list:
        """批次结论 → 每块板 → 未经修改的原始读数（含哈希）。"""
        batch = self._batch_or_404(batch_id)
        out = []
        for p in self.get_batch(batch_id)["plates"]:
            out.append(
                {
                    "plate_id": p["id"],
                    "plate_code": p["plate_code"],
                    "content_hash": p["content_hash"],
                    "raw_import": p["raw_imports"][0] if p["raw_imports"] else None,
                    "readings": p["readings"],
                }
            )
        return out


def _failed_fit(assigned: float, error: str) -> dict:
    empty_curve = {
        "converged": False,
        "A": None,
        "B": None,
        "ec50": None,
        "C": None,
        "D": None,
        "sse": None,
        "df": 0,
        "residual_sd": None,
        "warnings": ["FATAL:" + error],
        "quantitative_range": [None, None],
        "pooled_cv_percent": None,
        "max_cv_percent": None,
        "min_recovery_percent": None,
        "max_recovery_percent": None,
        "dose_stats": [],
        "se": {},
    }
    return {
        "converged": False,
        "iterations": 0,
        "fatal_error": error,
        "common_slope": None,
        "assigned_potency": assigned,
        "relative_potency": None,
        "ratio_ec50": None,
        "ci95": [None, None],
        "ci_half_width_percent": None,
        "parallelism": {
            "method": "F-test (common vs free slope)",
            "f_statistic": None,
            "df1": 1,
            "df2": 0,
            "p_value": None,
            "passed": False,
        },
        "standard": empty_curve,
        "sample": {**empty_curve, "role": "sample"},
        "warnings": ["FATAL:" + error],
    }


def _float_default(o):
    if isinstance(o, float):
        if o != o:  # NaN
            return None
        if o in (float("inf"), float("-inf")):
            return None
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"不可序列化 {type(o)!r}")
