"""文件级 SQLite 持久化测试：落盘、重开、触发器依然生效。"""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

from potency.db import connect, seed
from potency.service import Service
from tests.datafactory import make_wells, plate_payload


class TestFilePersistence(unittest.TestCase):
    def test_reopen_database_preserves_state_and_immutability(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            clock = lambda: datetime(2026, 9, 20, tzinfo=timezone.utc)
            conn = connect(path)
            seed(conn)
            svc = Service(conn, clock=clock)
            plate = svc.import_plate(
                plate_payload("DISK-1", make_wells(sample_ec50=0.85)), "analyst.li"
            )
            svc.create_analysis(plate["id"], "analyst.li")
            conn.commit()
            conn.close()

            # 重新打开：数据与状态完整保留
            conn2 = connect(path)
            svc2 = Service(conn2, clock=clock)
            again = svc2.get_plate(plate["id"])
            self.assertEqual(again["plate_code"], "DISK-1")
            self.assertEqual(again["status"], "analyzed")
            analyses = svc2.list_plate_analyses(plate["id"])
            self.assertEqual(len(analyses), 1)
            self.assertTrue(analyses[0]["evaluation"]["valid"])

            # 重开后不可变触发器仍然拦截
            with self.assertRaises(sqlite3.IntegrityError):
                conn2.execute(
                    "UPDATE plate_readings SET reading=1.0 WHERE plate_id=?",
                    (plate["id"],),
                )
            conn2.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
