"""SQLite 持久化层。

设计原则：
- 追加式：原始读数、分析结果、排除记录、结论一旦写入不修改、不删除；
  仅有的 UPDATE 是工作流状态推进（批次状态、复测申请状态、测定被取代标记、
  排除记录补登前后对照），且每次推进都写审计日志。
- 复杂对象以 JSON 文本存储，写入前经 sanitize 处理（非有限浮点 → null）。
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS method_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  fit_config TEXT NOT NULL,
  ruleset TEXT NOT NULL,
  combination_strategy TEXT NOT NULL,
  release_low_pct REAL NOT NULL,
  release_high_pct REAL NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (code, version)
);
CREATE TABLE IF NOT EXISTS standards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lot TEXT NOT NULL UNIQUE,
  assigned_potency REAL NOT NULL,
  unit TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL UNIQUE,
  method_version_id INTEGER NOT NULL REFERENCES method_versions(id),
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retest_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES batches(id),
  requested_by TEXT NOT NULL,
  requester_role TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL,
  approved_by TEXT,
  approver_role TEXT,
  created_at TEXT NOT NULL,
  decided_at TEXT
);
CREATE TABLE IF NOT EXISTS plates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES batches(id),
  method_version_id INTEGER NOT NULL REFERENCES method_versions(id),
  standard_id INTEGER NOT NULL REFERENCES standards(id),
  plate_label TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  payload TEXT NOT NULL,
  imported_by TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  retest_request_id INTEGER REFERENCES retest_requests(id)
);
CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  plate_id INTEGER NOT NULL REFERENCES plates(id),
  version INTEGER NOT NULL,
  method_version_id INTEGER NOT NULL REFERENCES method_versions(id),
  ruleset_snapshot TEXT NOT NULL,
  fit_config_snapshot TEXT NOT NULL,
  exclusion_ids TEXT NOT NULL,
  result TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (plate_id, version)
);
CREATE TABLE IF NOT EXISTS exclusions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  plate_id INTEGER NOT NULL REFERENCES plates(id),
  well TEXT NOT NULL,
  operator TEXT NOT NULL,
  reason TEXT NOT NULL,
  before_analysis_id INTEGER REFERENCES analyses(id),
  after_analysis_id INTEGER REFERENCES analyses(id),
  before_potency TEXT,
  after_potency TEXT,
  delta_potency TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (plate_id, well)
);
CREATE TABLE IF NOT EXISTS determinations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES batches(id),
  analysis_id INTEGER NOT NULL REFERENCES analyses(id),
  series TEXT NOT NULL,
  log_potency REAL,
  potency_pct REAL,
  se REAL,
  df INTEGER,
  valid INTEGER NOT NULL,
  superseded INTEGER NOT NULL DEFAULT 0,
  rule_hits TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conclusions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES batches(id),
  kind TEXT NOT NULL,
  strategy TEXT NOT NULL,
  determination_ids TEXT NOT NULL,
  n_determinations INTEGER NOT NULL,
  combined_potency_pct REAL,
  ci_lower_pct REAL,
  ci_upper_pct REAL,
  outcome TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  actor TEXT NOT NULL,
  detail TEXT,
  created_at TEXT NOT NULL
);
"""

# 需要按 JSON 解析的列
JSON_COLUMNS = {
    "method_versions": {"fit_config", "ruleset"},
    "plates": {"payload"},
    "analyses": {"ruleset_snapshot", "fit_config_snapshot", "exclusion_ids", "result"},
    "exclusions": {"before_potency", "after_potency", "delta_potency"},
    "determinations": {"rule_hits"},
    "conclusions": {"determination_ids"},
    "audit_log": {"detail"},
}


def sanitize(obj: Any) -> Any:
    """递归地把非有限浮点替换为 None，保证可序列化为合法 JSON。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """确定性 JSON 序列化：键排序、紧凑分隔符。用于内容哈希与结果比对。"""
    return json.dumps(
        sanitize(obj), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


class Store:
    """薄封装：一个 Store 持有一个 SQLite 连接。"""

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def insert(self, table: str, row: dict) -> int:
        cols = list(row.keys())
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        cur = self.conn.execute(sql, [self._encode(table, c, row[c]) for c in cols])
        self.conn.commit()
        return int(cur.lastrowid)

    def update(self, table: str, row_id: int, changes: dict) -> None:
        sets = ", ".join(f"{c} = ?" for c in changes)
        params = [self._encode(table, c, v) for c, v in changes.items()]
        self.conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*params, row_id])
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def one(self, table: str, where: str, params: tuple = ()) -> dict | None:
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE {where}", params)
        row = cur.fetchone()
        return self._decode(table, row) if row else None

    def by_id(self, table: str, row_id: int) -> dict | None:
        return self.one(table, "id = ?", (row_id,))

    def all(self, table: str, where: str = "1=1", params: tuple = (), order: str = "id") -> list[dict]:
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY {order}", params)
        return [self._decode(table, r) for r in cur.fetchall()]

    @staticmethod
    def _encode(table: str, col: str, value: Any) -> Any:
        if col in JSON_COLUMNS.get(table, set()) and value is not None and not isinstance(value, str):
            return canonical_json(value)
        return value

    @staticmethod
    def _decode(table: str, row: sqlite3.Row) -> dict:
        out = dict(row)
        for col in JSON_COLUMNS.get(table, set()):
            if out.get(col) is not None and isinstance(out[col], str):
                out[col] = json.loads(out[col])
        return out
