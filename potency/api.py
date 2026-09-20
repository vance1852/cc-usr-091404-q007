"""JSON HTTP API（标准库 http.server，无第三方依赖）。

鉴权用 ``X-User`` 请求头携带用户名（演示用，角色由数据库 users 表决定）。
所有错误统一为 ``{"error": {"code", "message"}}``。

路由
----
GET  /health
GET  /rulesets
POST /plates                          导板（重复导入返回 409 + 既有板信息）
GET  /plates/{id}
GET  /plates/{id}/analyses
POST /plates/{id}/analyses            新增分析版本（body: exclusions, note）
GET  /analyses/{id}
GET  /analyses/{a}/compare/{b}
POST /plates/{id}/lock
POST /plates/{id}/retest-requests
GET  /retest-requests/{id}
POST /retest-requests/{id}/decision
POST /batches
GET  /batches/{id}
POST /batches/{id}/plates
POST /batches/{id}/conclude
GET  /batches/{id}/trace              结论追溯到未经修改的原始读数
GET  /audit?entity=&id=
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .db import connect, seed
from .service import DuplicateImport, Service, ServiceError


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PotencyReview/1.0"

        # -- 工具 ----------------------------------------------------------
        def _send_json(self, obj, status: int = 200):
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, exc: ServiceError):
            payload = {"error": {"code": exc.code, "message": str(exc)}}
            if isinstance(exc, DuplicateImport):
                payload["error"]["existing_plate_id"] = exc.existing_plate_id
                payload["error"]["existing_plate_code"] = exc.existing_plate_code
                payload["error"]["content_hash"] = exc.content_hash
            self._send_json(payload, exc.status)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ServiceError("请求体不是合法 JSON", "BAD_JSON", 400)
            if not isinstance(data, dict):
                raise ServiceError("请求体必须是 JSON 对象", "BAD_JSON", 400)
            return data

        def _actor(self, body: dict) -> str:
            actor = self.headers.get("X-User") or body.pop("_actor", None)
            if not actor:
                raise ServiceError("缺少 X-User 请求头", "NO_ACTOR", 401)
            return actor

        def log_message(self, fmt, *args):  # 静默，测试输出干净
            return

        # -- 入口 ----------------------------------------------------------
        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method):
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                qs = parse_qs(parsed.query)
                self._route(method, path, qs)
            except ServiceError as exc:
                self._send_error(exc)
            except Exception as exc:  # noqa: BLE001 - API 边界兜底
                self._send_json(
                    {"error": {"code": "INTERNAL", "message": repr(exc)}}, 500
                )

        def _route(self, method, path, qs):
            s = service

            if method == "GET" and path == "/health":
                return self._send_json({"status": "ok"})

            if method == "GET" and path == "/rulesets":
                from .rules import DEFAULT_RULES

                return self._send_json(
                    {
                        "rulesets": [
                            {**r.to_dict(),
                             "effective_from": r.effective_from.isoformat()}
                            for r in DEFAULT_RULES
                        ]
                    }
                )

            m = re.fullmatch(r"/plates/(\d+)", path)
            if method == "POST" and path == "/plates":
                body = self._read_json()
                actor = self._actor(body)
                return self._send_json(s.import_plate(body, actor), 201)
            if method == "GET" and m:
                return self._send_json(s.get_plate(int(m.group(1))))

            m = re.fullmatch(r"/plates/(\d+)/analyses", path)
            if m and method == "GET":
                return self._send_json(
                    {"analyses": s.list_plate_analyses(int(m.group(1)))}
                )
            if m and method == "POST":
                body = self._read_json()
                actor = self._actor(body)
                result = s.create_analysis(
                    int(m.group(1)),
                    actor,
                    exclusions=body.get("exclusions", []),
                    note=body.get("note", ""),
                )
                return self._send_json(result, 201)

            m = re.fullmatch(r"/plates/(\d+)/lock", path)
            if method == "POST" and m:
                body = self._read_json()
                actor = self._actor(body)
                return self._send_json(
                    s.lock_first_round(
                        int(m.group(1)), actor, body.get("analysis_version")
                    )
                )

            m = re.fullmatch(r"/plates/(\d+)/retest-requests", path)
            if method == "POST" and m:
                body = self._read_json()
                actor = self._actor(body)
                reason = body.get("reason", "")
                return self._send_json(
                    s.create_retest_request(int(m.group(1)), actor, reason), 201
                )

            m = re.fullmatch(r"/analyses/(\d+)", path)
            if method == "GET" and m:
                return self._send_json(s.get_analysis(int(m.group(1))))

            m = re.fullmatch(r"/analyses/(\d+)/compare/(\d+)", path)
            if method == "GET" and m:
                return self._send_json(
                    s.compare_analyses(int(m.group(1)), int(m.group(2)))
                )

            m = re.fullmatch(r"/retest-requests/(\d+)", path)
            if method == "GET" and m:
                return self._send_json(s.get_retest_request(int(m.group(1))))
            if method == "POST" and m:
                body = self._read_json()
                actor = self._actor(body)
                return self._send_json(
                    s.decide_retest(
                        int(m.group(1)),
                        actor,
                        body.get("decision", ""),
                        body.get("comment", ""),
                    )
                )

            if method == "POST" and path == "/batches":
                body = self._read_json()
                actor = self._actor(body)
                if not body.get("batch_code"):
                    raise ServiceError("batch_code 必填", "BAD_PAYLOAD")
                return self._send_json(
                    s.create_batch(
                        body["batch_code"],
                        body.get("product", ""),
                        body.get("strategy_code", "MEAN_ALL_VALID"),
                        actor,
                    ),
                    201,
                )

            m = re.fullmatch(r"/batches/(\d+)", path)
            if method == "GET" and m:
                return self._send_json(s.get_batch(int(m.group(1))))

            m = re.fullmatch(r"/batches/(\d+)/plates", path)
            if method == "POST" and m:
                body = self._read_json()
                actor = self._actor(body)
                if body.get("plate_id") is None:
                    raise ServiceError("plate_id 必填", "BAD_PAYLOAD")
                return self._send_json(
                    s.add_plate_to_batch(
                        int(m.group(1)), int(body["plate_id"]), actor
                    )
                )

            m = re.fullmatch(r"/batches/(\d+)/conclude", path)
            if method == "POST" and m:
                body = self._read_json()
                actor = self._actor(body)
                return self._send_json(s.conclude_batch(int(m.group(1)), actor))

            m = re.fullmatch(r"/batches/(\d+)/trace", path)
            if method == "GET" and m:
                return self._send_json(
                    {"sources": s.trace_batch_readings(int(m.group(1)))}
                )

            if method == "GET" and path == "/audit":
                return self._send_json(
                    {
                        "entries": s.list_audit(
                            entity=qs.get("entity", [None])[0],
                            entity_id=int(qs["id"][0]) if "id" in qs else None,
                        )
                    }
                )

            self._send_json(
                {"error": {"code": "NOT_FOUND", "message": f"无此路由: {method} {path}"}},
                404,
            )

    return Handler


def build_server(db_path: str = ":memory:", host: str = "127.0.0.1", port: int = 8080):
    conn = connect(db_path)
    seed(conn)
    service = Service(conn)
    return HTTPServer((host, port), make_handler(service))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="效价会审 API 服务")
    parser.add_argument("--db", default="potency_review.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = build_server(args.db, args.host, args.port)
    print(f"效价会审服务运行于 http://{args.host}:{args.port} (db={args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
