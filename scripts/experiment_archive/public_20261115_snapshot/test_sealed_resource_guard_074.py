import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from sealed_resource_guard_074 import bytes_without_symlinks, ResourceGuard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.now = 0.
        self.calls = 0
        def census(root):
            self.calls += 1
            return 20*2**30
        self.guard = ResourceGuard(self.path, self.path, census=census,
            clock=lambda: self.now, disk_usage=lambda p: SimpleNamespace(free=30*2**30))

    def tearDown(self):
        self.temp.cleanup()

    def test_periodic_global_census(self):
        self.guard(0, 100)
        self.now = 1
        self.guard(0, 100)
        self.assertEqual(self.calls, 1)
        self.now = 31
        self.guard(0, 100)
        self.assertEqual(self.calls, 2)

    def test_time_budget_unchanged(self):
        self.now = 101
        with self.assertRaises(TimeoutError):
            self.guard(0, 100)
        self.assertEqual(self.calls, 0)

    def test_low_disk_stops_each_call(self):
        self.guard.disk_usage = lambda p: SimpleNamespace(free=19*2**30)
        with self.assertRaises(OSError):
            self.guard(0, 100)

    def test_conservative_global_limit(self):
        self.guard.census = lambda root: 25*2**30-32*2**20
        with self.assertRaises(OSError):
            self.guard(0, 100)

    def test_additional_stage_limit(self):
        with (self.path/'sparse_test_file').open('wb') as f:
            f.truncate(64*2**20)
        with self.assertRaises(OSError):
            self.guard(0, 100)

    def test_symlinks_are_not_counted_as_new_output(self):
        (self.path/'real').write_bytes(b'123')
        (self.path/'link').symlink_to(self.path/'real')
        self.assertEqual(bytes_without_symlinks(self.path), 3)

    def test_forced_stage_end_census(self):
        self.guard(0, 100)
        self.guard(0, 100, force=True)
        self.assertEqual(self.calls, 2)


if __name__ == '__main__':
    unittest.main()
