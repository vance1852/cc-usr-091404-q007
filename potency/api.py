"""效价检测会审系统 HTTP API。

运行：uvicorn potency.api:app --port 8000
数据库路径由环境变量 POTENCY_DB_PATH 指定（默认 ./potency.db）。
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import DuplicatePlateError, NotFoundError, PotencyError, ValidationError, WorkflowError
from .services import PotencyService


# ---------------------------------------------------------------- 请求模型
class MethodVersionIn(BaseModel):
    code: str
    name: str
    version: str
    fit_config: dict | None = None
    ruleset: dict | None = None
    combination_strategy: Literal["mean", "inverse_variance", "median"] = "mean"
    release_low_pct: float = 80.0
    release_high_pct: float = 125.0
    actor: str = "system"


class StandardIn(BaseModel):
    lot: str
    assigned_potency: float = 1.0
    unit: str = "U/mL"
    actor: str = "system"


class BatchIn(BaseModel):
    code: str
    method_version_id: int
    actor: str = "system"


class SeriesDefIn(BaseModel):
    kind: Literal["standard", "sample"]
    concentrations: list[float] = Field(min_length=1)


class WellIn(BaseModel):
    well: str
    series: str
    level: int = Field(ge=0)
    response: float


class PlateIn(BaseModel):
    batch_code: str
    plate_label: str
    standard_lot: str
    series: dict[str, SeriesDefIn]
    wells: list[WellIn] = Field(min_length=1)
    imported_by: str
    retest_request_id: int | None = None


class ActorIn(BaseModel):
    actor: str


class ExclusionIn(BaseModel):
    well: str
    operator: str
    reason: str


class RoleActionIn(BaseModel):
    actor: str
    role: str


class RetestRequestIn(BaseModel):
    actor: str
    role: str
    reason: str


class RetestDecisionIn(BaseModel):
    actor: str
    role: str
    approve: bool = True


# ---------------------------------------------------------------- 应用工厂
def create_app(db_path: str = ":memory:", clock=None) -> FastAPI:
    app = FastAPI(title="效价检测会审系统", version="0.1.0")
    service = PotencyService(db_path=db_path, clock=clock)
    app.state.service = service

    @app.exception_handler(NotFoundError)
    async def _404(_: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(DuplicatePlateError)
    async def _409(_: Request, exc: DuplicatePlateError):
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "existing_plate_id": exc.existing_plate_id},
        )

    @app.exception_handler(WorkflowError)
    async def _400(_: Request, exc: WorkflowError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def _422(_: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # ---------------- 主数据 ----------------
    @app.post("/method-versions", status_code=201)
    def create_method_version(body: MethodVersionIn):
        return service.create_method_version(**body.model_dump())

    @app.get("/method-versions")
    def list_method_versions():
        return service.list_method_versions()

    @app.get("/method-versions/{method_id}")
    def get_method_version(method_id: int):
        return service.get_method_version(method_id)

    @app.post("/standards", status_code=201)
    def create_standard(body: StandardIn):
        return service.create_standard(**body.model_dump())

    @app.get("/standards/{standard_id}")
    def get_standard(standard_id: int):
        return service.get_standard(standard_id)

    @app.post("/batches", status_code=201)
    def create_batch(body: BatchIn):
        return service.create_batch(**body.model_dump())

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: int):
        return service.get_batch(batch_id)

    @app.get("/batches/{batch_id}/determinations")
    def list_determinations(batch_id: int, include_superseded: bool = False):
        return service.list_determinations(batch_id, include_superseded=include_superseded)

    @app.get("/batches/{batch_id}/conclusions")
    def list_conclusions(batch_id: int):
        return service.list_conclusions(batch_id)

    # ---------------- 板与分析 ----------------
    @app.post("/plates", status_code=201)
    def import_plate(body: PlateIn):
        return service.import_plate(**body.model_dump())

    @app.get("/plates/lookup/{content_hash}")
    def lookup_plate(content_hash: str):
        found = service.find_duplicate(content_hash)
        if found is None:
            raise NotFoundError(f"未找到内容哈希为 {content_hash} 的板")
        return found

    @app.get("/plates/{plate_id}")
    def get_plate(plate_id: int):
        return service.get_plate(plate_id)

    @app.get("/batches/{batch_id}/plates")
    def list_plates(batch_id: int):
        return service.list_plates(batch_id)

    @app.post("/plates/{plate_id}/analyze", status_code=201)
    def analyze_plate(plate_id: int, body: ActorIn):
        return service.analyze_plate(plate_id, actor=body.actor)

    @app.get("/plates/{plate_id}/analyses")
    def list_analyses(plate_id: int):
        return service.list_analyses(plate_id)

    @app.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: int):
        return service.get_analysis(analysis_id)

    @app.post("/analyses/{analysis_id}/recompute")
    def recompute_analysis(analysis_id: int):
        return service.recompute_analysis(analysis_id)

    # ---------------- 排除孔 ----------------
    @app.post("/plates/{plate_id}/exclusions", status_code=201)
    def exclude_well(plate_id: int, body: ExclusionIn):
        return service.exclude_well(plate_id, **body.model_dump())

    @app.get("/plates/{plate_id}/exclusions")
    def list_exclusions(plate_id: int):
        return service.list_exclusions(plate_id)

    # ---------------- 复测会审流程 ----------------
    @app.post("/batches/{batch_id}/first-round/lock", status_code=201)
    def lock_first_round(batch_id: int, body: RoleActionIn):
        return service.lock_first_round(batch_id, actor=body.actor, role=body.role)

    @app.post("/batches/{batch_id}/retest-requests", status_code=201)
    def request_retest(batch_id: int, body: RetestRequestIn):
        return service.request_retest(batch_id, **body.model_dump())

    @app.post("/retest-requests/{request_id}/decision")
    def decide_retest(request_id: int, body: RetestDecisionIn):
        return service.decide_retest(request_id, **body.model_dump())

    @app.get("/retest-requests/{request_id}")
    def get_retest_request(request_id: int):
        return service.get_retest_request(request_id)

    @app.post("/batches/{batch_id}/conclude", status_code=201)
    def conclude_batch(batch_id: int, body: RoleActionIn):
        return service.conclude_batch(batch_id, actor=body.actor, role=body.role)

    # ---------------- 追溯 ----------------
    @app.get("/batches/{batch_id}/trace")
    def trace_batch(batch_id: int):
        return service.trace_batch(batch_id)

    return app


app = create_app(os.environ.get("POTENCY_DB_PATH", "potency.db"))
