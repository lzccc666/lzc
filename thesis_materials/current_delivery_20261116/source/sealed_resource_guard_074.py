"""仅资源检查调度：每次检查物理盘/当前阶段写入，每30秒核对全目录。"""
from pathlib import Path
import shutil
import time


def bytes_without_symlinks(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*')
               if not p.is_symlink() and p.is_file())


class ResourceGuard:
    def __init__(self, root, stage_output, *, census=bytes_without_symlinks,
                 clock=time.monotonic, disk_usage=shutil.disk_usage):
        self.root = Path(root)
        self.stage_output = Path(stage_output)
        self.census = census
        self.clock = clock
        self.disk_usage = disk_usage
        self.last_scan = None
        self.global_bytes = None
        self.scans = 0
        self.interval_s = 30.
        self.reserve_bytes = 64*2**20
        self.global_limit = 25*2**30
        self.stage_limit = 64*2**20

    def __call__(self, begun, budget_s, *, force=False):
        now = self.clock()
        if now-begun > budget_s:
            raise TimeoutError('达到原冻结阶段时间预算，保留前缀')
        if self.disk_usage(self.root).free < 20*2**30:
            raise OSError('物理空闲少于原冻结20GiB')
        if bytes_without_symlinks(self.stage_output) >= self.stage_limit:
            raise OSError('本执行分支新增输出达到额外64MiB约束')
        if force or self.last_scan is None or now-self.last_scan >= self.interval_s:
            self.global_bytes = self.census(self.root)
            self.last_scan = self.clock()
            self.scans += 1
        if self.global_bytes+self.reserve_bytes >= self.global_limit:
            raise OSError('全目录计数及保守64MiB预留达到原冻结25GiB上限')
