"""API 端到端测试：完整会审流程走 HTTP 接口。"""
import pytest
from fastapi.testclient import TestClient

from potency.api import create_app
from tests.conftest import FIXED_TIME, make_plate


@pytest.fixture()
def client():
    app = create_app(db_path=":memory:", clock=lambda: FIXED_TIME)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def seeded(client: TestClient):
    r = client.post("/method-versions", json={
        "code": "CYTO-POT", "name": "细胞法效价测定", "version": "1.0",
        "combination_strategy": "mean", "release_low_pct": 80.0, "release_high_pct": 125.0,
        "actor": "qa.wang",
    })
    assert r.status_code == 201, r.text
    method_id = r.json()["id"]
    r = client.post("/standards", json={"lot": "STD-01", "assigned_potency": 1.0, "actor": "qa.wang"})
    assert r.status_code == 201
    r = client.post("/batches", json={"code": "B001", "method_version_id": method_id, "actor": "qa.wang"})
    assert r.status_code == 201
    return {"method_id": method_id, "batch_id": r.json()["id"]}


def import_plate(client: TestClient, **kwargs):
    payload = make_plate(**kwargs)
    return client.post("/plates", json=payload)


class TestEndToEnd:
    def test_full_review_cycle(self, client, seeded):
        # 1. 导入首轮板并分析：78%，低于放行限度
        r = import_plate(client, potency=0.78)
        assert r.status_code == 201, r.text
        plate1 = r.json()
        assert plate1["content_hash"]
        r = client.post(f"/plates/{plate1['id']}/analyze", json={"actor": "analyst.chen"})
        assert r.status_code == 201, r.text
        analysis1 = r.json()
        assert analysis1["result"]["samples"]["S1"]["valid"]
        potency1 = analysis1["result"]["samples"]["S1"]["potency"]["potency_pct"]
        assert potency1 == pytest.approx(78.0, abs=2.0)

        # 2. 重复导入同一内容 → 409
        r = import_plate(client, potency=0.78)
        assert r.status_code == 409
        assert r.json()["existing_plate_id"] == plate1["id"]

        # 3. 复算验证确定性
        r = client.post(f"/analyses/{analysis1['id']}/recompute")
        assert r.status_code == 200
        assert r.json()["matches"]

        # 4. 锁定首轮结论（分析员角色无权 → 400）
        r = client.post(f"/batches/{seeded['batch_id']}/first-round/lock",
                        json={"actor": "analyst.chen", "role": "ANALYST"})
        assert r.status_code == 400
        r = client.post(f"/batches/{seeded['batch_id']}/first-round/lock",
                        json={"actor": "sup.li", "role": "SUPERVISOR"})
        assert r.status_code == 201, r.text
        assert r.json()["outcome"] == "FAIL"

        # 5. 复测申请与批准（不同人、不同角色）
        r = client.post(f"/batches/{seeded['batch_id']}/retest-requests",
                        json={"actor": "analyst.chen", "role": "ANALYST",
                              "reason": "首轮效价低于放行限度"})
        assert r.status_code == 201
        req_id = r.json()["id"]
        r = client.post(f"/retest-requests/{req_id}/decision",
                        json={"actor": "analyst.chen", "role": "SUPERVISOR", "approve": True})
        assert r.status_code == 400  # 本人不得批准
        r = client.post(f"/retest-requests/{req_id}/decision",
                        json={"actor": "sup.li", "role": "SUPERVISOR", "approve": True})
        assert r.status_code == 200
        assert r.json()["status"] == "APPROVED"

        # 6. 复测板导入与分析
        r = client.post("/plates", json={**make_plate(label="P-RT", potency=0.86),
                                         "retest_request_id": req_id})
        assert r.status_code == 201
        plate2 = r.json()
        r = client.post(f"/plates/{plate2['id']}/analyze", json={"actor": "analyst.chen"})
        assert r.status_code == 201

        # 7. 最终结论：两次测定全部纳入
        r = client.post(f"/batches/{seeded['batch_id']}/conclude",
                        json={"actor": "qa.wang", "role": "QA"})
        assert r.status_code == 201, r.text
        final = r.json()
        assert final["n_determinations"] == 2
        assert final["outcome"] == "PASS"
        assert final["combined_potency_pct"] == pytest.approx(82.0, abs=2.5)

        # 8. 追溯：结论 → 测定 → 分析 → 板 → 原始读数
        r = client.get(f"/batches/{seeded['batch_id']}/trace")
        assert r.status_code == 200
        trace = r.json()
        assert trace["integrity"]["all_raw_readings_intact"]
        assert len(trace["plates"]) == 2
        assert all(p["hash_verified"] for p in trace["plates"])

    def test_exclusion_via_api(self, client, seeded):
        payload = make_plate(potency=0.82)
        for w in payload["wells"]:
            if w["series"] == "S1" and w["level"] == 2 and w["well"].endswith("6"):
                w["response"] = round(w["response"] * 3.0, 6)
                bad_well = w["well"]
                break
        r = client.post("/plates", json=payload)
        plate = r.json()
        r = client.post(f"/plates/{plate['id']}/analyze", json={"actor": "analyst.chen"})
        assert not r.json()["result"]["samples"]["S1"]["valid"]

        r = client.post(f"/plates/{plate['id']}/exclusions",
                        json={"well": bad_well, "operator": "analyst.chen",
                              "reason": "复孔离散超差"})
        assert r.status_code == 201, r.text
        exc = r.json()
        assert exc["delta_potency"]["S1"] is not None

        r = client.get(f"/plates/{plate['id']}/exclusions")
        assert len(r.json()) == 1
        r = client.get(f"/plates/{plate['id']}/analyses")
        assert len(r.json()) == 2  # 排除前后各一版

    def test_404_and_validation(self, client, seeded):
        assert client.get("/plates/999").status_code == 404
        assert client.get("/batches/999").status_code == 404
        bad = make_plate()
        bad["wells"][0]["series"] = "GHOST"
        assert client.post("/plates", json=bad).status_code == 422

    def test_duplicate_lookup(self, client, seeded):
        r = import_plate(client, potency=0.8)
        plate = r.json()
        r = client.get(f"/plates/lookup/{plate['content_hash']}")
        assert r.status_code == 200
        assert r.json()["id"] == plate["id"]
        assert client.get("/plates/lookup/" + "0" * 64).status_code == 404
