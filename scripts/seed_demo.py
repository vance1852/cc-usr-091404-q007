#!/usr/bin/env python3
"""端到端演示：离散读数板的会审全过程。

场景对应需求：首算 77.1% 恰低于放行限（80%），排除一个离群孔后"变成"
80.8%。系统不删除失败版本，而是保留两个只追加版本、排孔前后差异、
首轮锁定、多角色复测审批，最终批次结论强制纳入谱系内全部测定。

运行：python3 scripts/seed_demo.py
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potency.db import connect, seed  # noqa: E402
from potency.service import Service  # noqa: E402
from tests.datafactory import make_wells, plate_payload  # noqa: E402


def line(t=""):
    print(t)


def main():
    db_path = os.environ.get("DEMO_DB", "demo_potency.db")
    if os.path.exists(db_path):
        os.unlink(db_path)
    conn = connect(db_path)
    seed(conn)
    clock_value = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)
    svc = Service(conn, clock=lambda: clock_value)

    line("=" * 72)
    line("效价会审演示 · 规则集 CELL_POTENCY v1.1（2026 年生效）")
    line("=" * 72)

    # --- 导入首轮板（含一个离群孔）---
    wells = make_wells(sample_ec50=1.0)
    for w in wells:
        if w["well"] == "E09":  # 样品 EC50 附近剂量孔，读数异常偏高
            w["reading"] = round(w["reading"] + 0.15, 6)
    plate = svc.import_plate(plate_payload("PLATE-A", wells), "analyst.li")
    line(f"[1] 导入板 PLATE-A（{len(plate['readings'])} 孔）"
         f"  标准品批号 {plate['standard_lot']}  内容哈希 {plate['content_hash'][:16]}…")

    # --- 首轮分析：低于限度 ---
    v1 = svc.create_analysis(plate["id"], "analyst.li", note="首轮：原始全孔")
    line(f"[2] 分析 v{v1['version_no']}：效价 "
         f"{v1['fit']['relative_potency']:.2f}（标示 100），"
         f"合格={v1['evaluation']['valid']}，放行={v1['evaluation']['release_passed']}")

    # --- 第二版：排孔 ---
    v2 = svc.create_analysis(
        plate["id"], "analyst.li",
        exclusions=[{"well": "E09", "operator": "analyst.li",
                     "reason": "复核发现移液气泡，主管现场签认"}],
        note="排除 E09 后重算",
    )
    ex = v2["exclusions"][0]
    line(f"[3] 分析 v{v2['version_no']}：排除 E09 后效价 "
         f"{v2['fit']['relative_potency']:.2f}；"
         f"该孔排除前 {ex['potency_before']:.2f} → 排除后 {ex['potency_after']:.2f}"
         f"（差异 {ex['delta_potency_pct']:+.2f}%），操作者/理由已留痕")

    # --- 重复导入识别 ---
    try:
        svc.import_plate(plate_payload("PLATE-A-COPY", wells), "analyst.li")
    except Exception as e:  # DuplicateImport
        line(f"[4] 重复导入被拦截：{e}")

    # --- 版本比较 ---
    cmp_ = svc.compare_analyses(v1["id"], v2["id"])
    line(f"[5] 版本比较：效价 {cmp_['potency']['a_percent']:.2f}% → "
         f"{cmp_['potency']['b_percent']:.2f}%，新增排孔 "
         f"{cmp_['exclusions_added_in_b']}，平行性 p 值 "
         f"{v1['fit']['parallelism']['p_value']:.3f} / "
         f"{v2['fit']['parallelism']['p_value']:.3f}")

    # --- 锁定首轮结论（以 v1 为准锁定，不允许只留有利的 v2）---
    locked = svc.lock_first_round(plate["id"], "supervisor.wang",
                                  analysis_version=v1["version_no"])
    line(f"[6] 主管锁定首轮结论：status={locked['status']}（基于 v1，"
         f"低于放行限）；此后板冻结，不能再新增分析版本")

    # --- 复测申请与多角色审批 ---
    req = svc.create_retest_request(plate["id"], "analyst.li",
                                    "首读低于限度且存在已签认离群孔，申请复测")
    line(f"[7] 复测申请 #{req['id']}：待 SUPERVISOR + QA 分别批准")
    svc.decide_retest(req["id"], "supervisor.wang", "approved", comment="同意，关注移液")
    line("    - 主管批准后仍 pending")
    svc.decide_retest(req["id"], "qa.zhao", "approved", comment="QA 同意，复测板须纳入批次")
    line("    - QA 批准后 status=approved")

    # --- 复测板 ---
    retest = svc.import_plate(
        plate_payload("PLATE-A-R1", make_wells(sample_ec50=0.85),
                      retest_request_id=req["id"]),
        "analyst.li",
    )
    rv = svc.create_analysis(retest["id"], "analyst.li", note="复测")
    svc.lock_first_round(retest["id"], "supervisor.wang")
    line(f"[8] 复测板 PLATE-A-R1（同谱系）：效价 "
         f"{rv['fit']['relative_potency']:.2f}，放行={rv['evaluation']['release_passed']}")

    # --- 批次：试图只挑好板被拒；纳入全部后结论 ---
    batch = svc.create_batch("B-2026-0920", "产品X", "MEAN_ALL_VALID", "analyst.li")
    svc.add_plate_to_batch(batch["id"], retest["id"], "analyst.li")
    try:
        svc.conclude_batch(batch["id"], "qa.zhao")
    except Exception as e:
        line(f"[9] 只纳入复测好板 → 拒绝：{e}")
    svc.add_plate_to_batch(batch["id"], plate["id"], "analyst.li")
    result = svc.conclude_batch(batch["id"], "qa.zhao")
    c = result["conclusion"]
    line(f"[10] 纳入全部 {c['n_plates']} 块板（算术平均）：组合效价 "
         f"{c['combined_potency_percent']:.2f}%，全部有效={c['all_plates_valid']}，"
         f"放行={c['released']}")

    # --- 追溯 ---
    sources = svc.trace_batch_readings(batch["id"])
    line(f"[11] 批次结论可追溯到 {len(sources)} 块板的未经修改原始读数：")
    for s in sources:
        line(f"     - 板#{s['plate_id']} 哈希 {s['content_hash'][:16]}… "
             f"{len(s['readings'])} 孔，原始文件 {s['raw_import']['source_name']}")
    line("=" * 72)
    line(f"完成。数据库文件：{db_path}（可用 python3 -m potency.api --db {db_path} 起服务复查）")


if __name__ == "__main__":
    main()
