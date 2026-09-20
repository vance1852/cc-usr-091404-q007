"""测试共用：确定性合成板数据（无随机数，结果可逐位复现）。"""

from __future__ import annotations

from potency.fitting import four_pl

DOSES = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
ROWS = "ABCDEFGH"


def _noise(well_index: int, dose_index: int, role: str) -> float:
    """确定性微小扰动（固定余弦序列，不使用随机数）。

    标准品与样品共用同一序列（忽略 role），保证两条曲线除 EC50 外
    统计上真正平行——用于平行性检验的阳性用例。
    """
    import math

    return 0.006 * math.cos((well_index + 1) * (dose_index + 1) * 0.9 + 0.31)


def make_wells(
    sample_ec50: float = 1.0,
    hill: float = 1.1,
    bottom: float = 0.2,
    top: float = 1.5,
    replicates: int = 2,
    noise: bool = True,
    sample_doses=None,
) -> list[dict]:
    """生成标准品（EC50=0.8）+ 样品（EC50 可调）的 96 孔式记录。

    布局：A-D 行标准品，E-H 行样品；每个剂量 ``replicates`` 个重复。
    """
    wells = []
    idx = 0
    for r in range(4):
        for di, d in enumerate(DOSES):
            for rep in range(replicates):
                row = ROWS[r]
                col = di * replicates + rep + 1
                y = four_pl(d, bottom, hill, 0.8, top)
                if noise:
                    y += _noise(idx, di, "standard")
                wells.append(
                    {"well": f"{row}{col:02d}", "role": "standard",
                     "dose": d, "reading": round(y, 6)}
                )
                idx += 1
    s_doses = sample_doses or DOSES
    idx = 0
    for r in range(4, 8):
        for di, d in enumerate(s_doses):
            for rep in range(replicates):
                row = ROWS[r]
                col = di * replicates + rep + 1
                y = four_pl(d, bottom, hill, sample_ec50, top)
                if noise:
                    y += _noise(idx, di, "sample")
                wells.append(
                    {"well": f"{row}{col:02d}", "role": "sample",
                     "dose": d, "reading": round(y, 6)}
                )
                idx += 1
    return wells


def plate_payload(
    plate_code: str,
    wells: list[dict],
    *,
    assigned_potency: float = 100.0,
    standard_lot: str = "STD-LOT-2026-01",
    sample_lot: str = "BATCH-2026-09",
    retest_request_id=None,
) -> dict:
    return {
        "plate_code": plate_code,
        "method_code": "CELL_ASSAY_4PL",
        "method_version": "2.3",
        "standard_lot": standard_lot,
        "sample_lot": sample_lot,
        "assigned_potency": assigned_potency,
        "source_name": f"{plate_code}.csv",
        "wells": wells,
        "retest_request_id": retest_request_id,
    }
