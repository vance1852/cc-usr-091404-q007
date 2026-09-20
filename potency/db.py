"""SQLite 持久化层。

关键完整性约束：

* ``plate_readings`` 与 ``raw_imports`` 由触发器禁止 UPDATE/DELETE ——
  原始读数一经写入不可修改，批次结论可一直追溯到未改动的原始值。
* 分析版本、排孔记录、审批记录同样只追加（append-only）。
* 所有时间戳由调用方注入（默认 UTC），测试可固定时钟。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

from .rules import DEFAULT_RULES

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('ANALYST','SUPERVISOR','QA','REVIEWER'))
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (code, version)
);

CREATE TABLE IF NOT EXISTS rule_sets (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL,
    version TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    release_low_pct REAL NOT NULL,
    release_high_pct REAL NOT NULL,
    rule_codes_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    UNIQUE (code, version)
);

CREATE TABLE IF NOT EXISTS combination_strategies (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY,
    plate_code TEXT NOT NULL,
    method_id INTEGER NOT NULL REFERENCES methods(id),
    standard_lot TEXT NOT NULL,
    sample_lot TEXT NOT NULL,
    assigned_potency REAL NOT NULL CHECK (assigned_potency > 0),
    dilution_json TEXT NOT NULL,
    layout_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('imported','analyzed','locked','invalid_locked','superseded')),
    content_hash TEXT NOT NULL,
    imported_by TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    analyzed_at TEXT,
    locked_at TEXT,
    locked_analysis_id INTEGER,
    retest_of_request_id INTEGER,
    lineage_key TEXT NOT NULL,
    UNIQUE (plate_code)
);

CREATE TABLE IF NOT EXISTS plate_readings (
    id INTEGER PRIMARY KEY,
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    well TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('standard','sample')),
    dose REAL NOT NULL CHECK (dose > 0),
    reading REAL NOT NULL,
    UNIQUE (plate_id, well)
);

CREATE TABLE IF NOT EXISTS raw_imports (
    id INTEGER PRIMARY KEY,
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    source_name TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    imported_by TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    version_no INTEGER NOT NULL,
    ruleset_id INTEGER NOT NULL REFERENCES rule_sets(id),
    fit_json TEXT NOT NULL,
    eval_json TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (plate_id, version_no)
);

CREATE TABLE IF NOT EXISTS analysis_exclusions (
    id INTEGER PRIMARY KEY,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id),
    well TEXT NOT NULL,
    role TEXT NOT NULL,
    dose REAL NOT NULL,
    reading REAL NOT NULL,
    operator TEXT NOT NULL,
    reason TEXT NOT NULL,
    potency_before REAL,
    potency_after REAL,
    delta_potency_pct REAL,
    excluded_at TEXT NOT NULL,
    UNIQUE (analysis_id, well)
);

CREATE TABLE IF NOT EXISTS retest_requests (
    id INTEGER PRIMARY KEY,
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    lineage_key TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected')),
    created_at TEXT NOT NULL,
    decided_at TEXT
);

CREATE TABLE IF NOT EXISTS retest_approvals (
    id INTEGER PRIMARY KEY,
    request_id INTEGER NOT NULL REFERENCES retest_requests(id),
    approver_user TEXT NOT NULL,
    approver_role TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approved','rejected')),
    comment TEXT NOT NULL DEFAULT '',
    decided_at TEXT NOT NULL,
    UNIQUE (request_id, approver_role)
);

CREATE TABLE IF NOT EXISTS retest_plates (
    request_id INTEGER NOT NULL REFERENCES retest_requests(id),
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    PRIMARY KEY (request_id, plate_id)
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    batch_code TEXT NOT NULL UNIQUE,
    product TEXT NOT NULL,
    strategy_id INTEGER NOT NULL REFERENCES combination_strategies(id),
    status TEXT NOT NULL CHECK (status IN ('open','concluded')),
    conclusion_json TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    concluded_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_plates (
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    plate_id INTEGER NOT NULL REFERENCES plates(id),
    added_at TEXT NOT NULL,
    PRIMARY KEY (batch_id, plate_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    entity TEXT NOT NULL,
    entity_id INTEGER,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    at TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_readings_plate ON plate_readings(plate_id);
CREATE INDEX IF NOT EXISTS idx_analyses_plate ON analyses(plate_id);
CREATE INDEX IF NOT EXISTS idx_plates_lineage ON plates(lineage_key);
CREATE INDEX IF NOT EXISTS idx_retest_req ON retest_requests(lineage_key);

-- 原始数据不可变 ----------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_readings_no_update
BEFORE UPDATE ON plate_readings
BEGIN
    SELECT RAISE(ABORT, 'plate_readings 不可修改：原始读数受保护');
END;
CREATE TRIGGER IF NOT EXISTS trg_readings_no_delete
BEFORE DELETE ON plate_readings
BEGIN
    SELECT RAISE(ABORT, 'plate_readings 不可删除：原始读数受保护');
END;
CREATE TRIGGER IF NOT EXISTS trg_imports_no_update
BEFORE UPDATE ON raw_imports
BEGIN
    SELECT RAISE(ABORT, 'raw_imports 不可修改');
END;
CREATE TRIGGER IF NOT EXISTS trg_imports_no_delete
BEFORE DELETE ON raw_imports
BEGIN
    SELECT RAISE(ABORT, 'raw_imports 不可删除');
END;
CREATE TRIGGER IF NOT EXISTS trg_analyses_no_update
BEFORE UPDATE ON analyses
BEGIN
    SELECT RAISE(ABORT, 'analyses 只可追加，不可修改');
END;
CREATE TRIGGER IF NOT EXISTS trg_exclusions_no_update
BEFORE UPDATE ON analysis_exclusions
BEGIN
    SELECT RAISE(ABORT, 'analysis_exclusions 只可追加');
END;
CREATE TRIGGER IF NOT EXISTS trg_exclusions_no_delete
BEFORE DELETE ON analysis_exclusions
BEGIN
    SELECT RAISE(ABORT, 'analysis_exclusions 不可删除');
END;
"""


def connect(db_path: str = ":memory:") -> sqlite3.Connection:
    # check_same_thread=False：API 在单线程 HTTPServer 工作线程中使用
    # 在主线程创建的连接；ThreadingHTTPServer 部署时调用方需自行加锁。
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def seed(conn: sqlite3.Connection) -> None:
    """写入内置用户、规则集、示例方法与组合策略（幂等）。"""
    init_db(conn)
    users = [
        ("analyst.li", "李分析师", "ANALYST"),
        ("supervisor.wang", "王主管", "SUPERVISOR"),
        ("qa.zhao", "赵QA", "QA"),
        ("reviewer.chen", "陈会审人", "REVIEWER"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO users(username, display_name, role) VALUES (?,?,?)",
        users,
    )

    from .rules import _CONTEXT_BY_VERSION  # 复用随版参数

    for rs in DEFAULT_RULES:
        ctx = _CONTEXT_BY_VERSION.get(rs.version, {})
        conn.execute(
            """INSERT OR IGNORE INTO rule_sets
               (code, version, effective_from, description,
                release_low_pct, release_high_pct, rule_codes_json, context_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                rs.code,
                rs.version,
                rs.effective_from.isoformat(),
                rs.description,
                rs.release_low_pct,
                rs.release_high_pct,
                json.dumps([c.code for c in rs.checks], ensure_ascii=False),
                json.dumps(ctx, ensure_ascii=False, default=_json_default),
            ),
        )

    conn.execute(
        """INSERT OR IGNORE INTO methods(code, version, description, params_json, created_at)
           VALUES (?,?,?,?,?)""",
        (
            "CELL_ASSAY_4PL",
            "2.3",
            "细胞法信号-剂量 4PL 效价测定；共用斜率 EC50 比值法",
            json.dumps(
                {
                    "model": "4PL A+(D-A)/(1+(x/C)^B)",
                    "optimizer": "Levenberg-Marquardt",
                    "parallelism_test": "F-test common vs free slope",
                    "dilution_factor": 2,
                },
                ensure_ascii=False,
            ),
            now_iso(),
        ),
    )

    strategies = [
        (
            "MEAN_ALL_VALID",
            "纳入批次谱系内全部有效板，算术平均；任一板无效或组合值越限即不放行",
            {
                "aggregation": "mean",
                "all_plates_must_be_valid": True,
                "release_range_pct": [80.0, 125.0],
            },
        ),
        (
            "GMEAN_ALL_VALID",
            "纳入批次谱系内全部有效板，几何平均；任一板无效即不放行",
            {
                "aggregation": "geometric_mean",
                "all_plates_must_be_valid": True,
                "release_range_pct": [80.0, 125.0],
            },
        ),
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO combination_strategies(code, description, config_json)
           VALUES (?,?,?)""",
        [(c, d, json.dumps(cfg, ensure_ascii=False)) for c, d, cfg in strategies],
    )
    conn.commit()


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"不可序列化类型 {type(o)!r}")
