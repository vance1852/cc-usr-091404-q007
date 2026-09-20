"""HTTP API 端到端测试：真实起服 + urllib 客户端。"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer

from potency.api import make_handler
from potency.db import connect, seed
from potency.service import Service
from tests.datafactory import make_wells, plate_payload


class ApiClient:
    def __init__(self, base: str):
        self.base = base

    def call(self, method: str, path: str, body=None, user=None):
        url = self.base + path
        data = None
        headers = {"Content-Type": "application/json"}
        if user:
            headers["X-User"] = user
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                payload = resp.read().decode("utf-8")
                return resp.status, json.loads(payload) if payload else {}
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8")
            return e.code, json.loads(payload) if payload else {}


class ApiTestCase(unittest.TestCase):
    fixed_now = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)

    def setUp(self):
        conn = connect(":memory:")
        seed(conn)
        self.clock_value = self.fixed_now
        service = Service(conn, clock=lambda: self.clock_value)
        server = HTTPServer(("127.0.0.1", 0), make_handler(service))
        self.port = server.server_address[1]
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()
        self.server = server
        self.api = ApiClient(f"http://127.0.0.1:{self.port}")

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def test_health_and_rulesets(self):
        status, body = self.api.call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        status, body = self.api.call("GET", "/rulesets")
        self.assertEqual(status, 200)
        self.assertEqual([r["version"] for r in body["rulesets"]], ["1.0", "1.1"])

    def test_requires_user_header(self):
        status, body = self.api.call(
            "POST", "/plates", plate_payload("X", make_wells())
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "NO_ACTOR")

    def test_full_review_scenario_over_http(self):
        # 1) 首轮板（离群，低于限度）
        wells = make_wells(sample_ec50=1.0)
        for w in wells:
            if w["well"] == "E09":
                w["reading"] = round(w["reading"] + 0.15, 6)
        status, plate = self.api.call(
            "POST", "/plates", plate_payload("HTTP-P1", wells), user="analyst.li"
        )
        self.assertEqual(status, 201)
        pid = plate["id"]
        self.assertEqual(len(plate["readings"]), 112)

        # 2) 重复导入 → 409
        status, dup = self.api.call(
            "POST", "/plates", plate_payload("HTTP-P1-DUP", wells),
            user="analyst.li",
        )
        self.assertEqual(status, 409)
        self.assertEqual(dup["error"]["code"], "DUPLICATE_IMPORT")
        self.assertEqual(dup["error"]["existing_plate_id"], pid)

        # 3) 首轮分析（失败）与排孔分析（回升）两个版本
        status, a1 = self.api.call("POST", f"/plates/{pid}/analyses",
                                   {"note": "首轮"}, user="analyst.li")
        self.assertEqual(status, 201)
        self.assertLess(a1["fit"]["relative_potency"], 80.0)
        status, a2 = self.api.call(
            "POST", f"/plates/{pid}/analyses",
            {"exclusions": [{"well": "E09", "operator": "analyst.li",
                             "reason": "移液异常"}], "note": "排孔"},
            user="analyst.li",
        )
        self.assertEqual(status, 201)
        self.assertGreater(a2["fit"]["relative_potency"],
                           a1["fit"]["relative_potency"])

        # 4) 版本比较
        status, cmp_ = self.api.call(
            "GET", f"/analyses/{a1['id']}/compare/{a2['id']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(cmp_["exclusions_added_in_b"], ["E09"])

        # 5) 主管锁定首轮结论
        status, locked = self.api.call("POST", f"/plates/{pid}/lock",
                                       {}, user="supervisor.wang")
        self.assertEqual(status, 200)
        self.assertIn(locked["status"], ("locked", "invalid_locked"))

        # 6) 复测申请：需主管+QA 两角色
        status, req = self.api.call(
            "POST", f"/plates/{pid}/retest-requests",
            {"reason": "边缘失效，怀疑移液"}, user="analyst.li",
        )
        self.assertEqual(status, 201)
        rid = req["id"]
        status, req = self.api.call(
            "POST", f"/retest-requests/{rid}",
            {"decision": "approved"}, user="supervisor.wang",
        )
        self.assertEqual(req["status"], "pending")  # 单方批准仍挂起
        status, req = self.api.call(
            "POST", f"/retest-requests/{rid}",
            {"decision": "approved"}, user="qa.zhao",
        )
        self.assertEqual(status, 200)
        self.assertEqual(req["status"], "approved")

        # 7) 复测板同谱系导入、分析、锁定
        status, retest = self.api.call(
            "POST", "/plates",
            plate_payload("HTTP-P1-R1", make_wells(sample_ec50=0.85),
                          retest_request_id=rid),
            user="analyst.li",
        )
        self.assertEqual(status, 201)
        self.assertEqual(retest["lineage_key"], locked["lineage_key"])
        rpid = retest["id"]
        self.api.call("POST", f"/plates/{rpid}/analyses", {}, user="analyst.li")
        self.api.call("POST", f"/plates/{rpid}/lock", {}, user="supervisor.wang")

        # 8) 批次结论：必须纳入两块板
        status, batch = self.api.call(
            "POST", "/batches",
            {"batch_code": "HTTP-B1", "product": "产品X",
             "strategy_code": "MEAN_ALL_VALID"},
            user="analyst.li",
        )
        bid = batch["id"]
        # 只加复测板 → 结论被拒绝（禁止挑有利结果）
        self.api.call("POST", f"/batches/{bid}/plates",
                      {"plate_id": rpid}, user="analyst.li")
        status, err = self.api.call("POST", f"/batches/{bid}/conclude",
                                    {}, user="qa.zhao")
        self.assertEqual(status, 409)
        self.assertEqual(err["error"]["code"], "LINEAGE_INCOMPLETE")
        # 补齐首轮板 → 可结论
        self.api.call("POST", f"/batches/{bid}/plates",
                      {"plate_id": pid}, user="analyst.li")
        status, concluded = self.api.call("POST", f"/batches/{bid}/conclude",
                                          {}, user="qa.zhao")
        self.assertEqual(status, 200)
        self.assertEqual(concluded["status"], "concluded")
        self.assertEqual(concluded["conclusion"]["n_plates"], 2)

        # 9) 追溯到原始读数
        status, trace = self.api.call("GET", f"/batches/{bid}/trace")
        self.assertEqual(status, 200)
        self.assertEqual(len(trace["sources"]), 2)
        for src in trace["sources"]:
            self.assertIsNotNone(src["content_hash"])
            self.assertTrue(src["readings"])

    def test_unknown_route_404(self):
        status, body = self.api.call("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "NOT_FOUND")

    def test_malformed_and_missing_fields_are_400(self):
        # 非法 JSON
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/plates",
            data=b"{not-json",
            headers={"Content-Type": "application/json", "X-User": "analyst.li"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("应返回 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            self.assertEqual(json.loads(e.read())["error"]["code"], "BAD_JSON")

        # 缺字段
        status, body = self.api.call(
            "POST", "/batches", {"product": "X"}, user="analyst.li"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_PAYLOAD")

    def test_role_enforced_over_http(self):
        status, body = self.api.call(
            "POST", "/plates", plate_payload("RX", make_wells()), user="qa.zhao"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"]["code"], "FORBIDDEN_ROLE")


if __name__ == "__main__":
    unittest.main()
