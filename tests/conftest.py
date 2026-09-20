"""测试共享装置：确定性合成板数据生成。

所有"噪声"均为井位的确定性正弦扰动，不使用任何随机数，
保证测试在任何机器上结果一致。
"""
from __future__ import annotations

import math

import pytest

from potency.services import PotencyService

# 标准品曲线参数（下降型细胞法曲线）
STD_A, STD_B, STD_D = 2.0, 1.0, 0.1
STD_EC50 = 0.5
STD_LC = math.log10(STD_EC50)
CONCENTRATIONS = [0.01, 0.032, 0.1, 0.32, 1.0, 3.2, 10.0]
REPLICATES = 3

FIXED_TIME = "2026-09-20T08:00:00+00:00"


def curve_response(conc: float, A: float, B: float, logC: float, D: float) -> float:
    return D + (A - D) / (1.0 + 10.0 ** (B * (math.log10(conc) - logC)))


def make_series_wells(
    name: str,
    logC: float,
    A: float = STD_A,
    B: float = STD_B,
    D: float = STD_D,
    noise: float = 0.02,
    replicates: int = REPLICATES,
    row_offset: int = 0,
) -> list[dict]:
    """生成一个系列的复孔读数（确定性扰动）。"""
    wells = []
    for lvl, conc in enumerate(CONCENTRATIONS):
        base = curve_response(conc, A, B, logC, D)
        for rep in range(replicates):
            y = base * (1.0 + noise * math.sin(lvl * 2.7 + rep * 1.9 + row_offset + 0.3))
            row = chr(ord("A") + row_offset + lvl)
            wells.append({
                "well": f"{row}{rep + 1 + (3 if name != 'STD' else 0)}",
                "series": name,
                "level": lvl,
                "response": round(y, 6),
            })
    return wells


def make_plate(
    batch_code: str = "B001",
    label: str = "P-001",
    standard_lot: str = "STD-01",
    potency: float = 0.8,
    noise: float = 0.02,
    sample_B: float = STD_B,
    sample_name: str = "S1",
) -> dict:
    """构造一块完整板载荷：标准系列 + 一个样品系列。"""
    sample_logC = STD_LC - math.log10(potency)
    wells = make_series_wells("STD", STD_LC, noise=noise, row_offset=0)
    wells += make_series_wells(sample_name, sample_logC, B=sample_B, noise=noise, row_offset=0)
    return {
        "batch_code": batch_code,
        "plate_label": label,
        "standard_lot": standard_lot,
        "series": {
            "STD": {"kind": "standard", "concentrations": list(CONCENTRATIONS)},
            sample_name: {"kind": "sample", "concentrations": list(CONCENTRATIONS)},
        },
        "wells": wells,
        "imported_by": "analyst.chen",
    }


@pytest.fixture()
def service() -> PotencyService:
    clock_iter = iter(lambda t=FIXED_TIME: t, None)  # 固定时钟
    svc = PotencyService(db_path=":memory:", clock=lambda: FIXED_TIME)
    mv = svc.create_method_version(
        code="CYTO-POT", name="细胞法效价测定", version="1.0",
        combination_strategy="mean", release_low_pct=80.0, release_high_pct=125.0,
        actor="qa.wang",
    )
    svc.create_standard(lot="STD-01", assigned_potency=1.0, unit="U/mL", actor="qa.wang")
    svc._test_method_id = mv["id"]
    return svc


@pytest.fixture()
def batch(service: PotencyService) -> dict:
    return service.create_batch(code="B001", method_version_id=service._test_method_id, actor="qa.wang")


def import_and_analyze(service: PotencyService, plate: dict) -> tuple[dict, dict]:
    """导入板并立即分析，返回 (plate, analysis)。"""
    imported = service.import_plate(**plate)
    analysis = service.analyze_plate(imported["id"], actor=plate["imported_by"])
    return imported, analysis
