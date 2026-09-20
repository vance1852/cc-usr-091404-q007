"""测试夹具：内存数据库 + 固定时钟。"""

import unittest
from datetime import datetime, timezone

from potency.db import connect, seed
from potency.service import Service


class ServiceTestCase(unittest.TestCase):
    """每个用例独立内存库，时钟固定在 2026-09-20（规则集 v1.1 生效期）。"""

    fixed_now = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)

    def setUp(self):
        self.conn = connect(":memory:")
        seed(self.conn)
        self.clock_value = self.fixed_now
        self.service = Service(self.conn, clock=lambda: self.clock_value)

    def tearDown(self):
        self.conn.close()

    def set_clock(self, dt: datetime):
        self.clock_value = dt

    # -- 常用快捷流程 -----------------------------------------------------
    def import_good_plate(self, code="PL-001", **wells_kw):
        from tests.datafactory import make_wells, plate_payload

        payload = plate_payload(code, make_wells(**wells_kw))
        return self.service.import_plate(payload, "analyst.li")

    def analyze_and_lock(self, plate_id, exclusions=None, actor_lock="supervisor.wang"):
        self.service.create_analysis(plate_id, "analyst.li", exclusions=exclusions)
        return self.service.lock_first_round(plate_id, actor_lock)
