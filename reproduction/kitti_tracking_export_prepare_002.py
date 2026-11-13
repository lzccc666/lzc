"""新连续KITTI业务源码的独立导出；保持原索引与历史产物。"""

from datetime import datetime, timezone
import json
import re
import shutil
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

root = OUTPUT / "source_bundle/github_sync_001"
previous = json.loads((root / "incremental_export_manifest_017.json").read_text())
names = {r["path"] for r in previous["source_files"]}
names.update(
    [
        "conceptgraph/completion/kitti_tracking_geometry.py",
        "conceptgraph/completion/kitti_tracking_experiment.py",
        "conceptgraph/completion/kitti_tracking_preview.py",
        "tests/test_completion_kitti_tracking_geometry.py",
        "tests/test_completion_kitti_tracking_experiment.py",
    ]
)
assert len(names) == 390
export = root / "checkout"
rows = []
secret = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
assert digest(PROJECT / ".git/index") == previous["original_checkout_index_sha256"]
for name in sorted(names):
    source = PROJECT / name
    target = export / name
    assert not secret.search(source.read_bytes()), name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert digest(source) == digest(target)
    rows.append(
        dict(
            path=name,
            sha256=digest(source),
            bytes=source.stat().st_size,
            executable=bool(source.stat().st_mode & 0o111),
        )
    )
value = dict(previous)
value.update(
    created_utc=datetime.now(timezone.utc).isoformat(),
    source_files=rows,
    current_sources_published=False,
    source_bytes_match=True,
    KITTI_tracking="固定90帧连续车载开发诊断；不能计入封存或导航到达",
    goal_complete=False,
)
dump(root / "incremental_export_manifest_018.json", value)
print(
    json.dumps(
        dict(
            source_files=len(rows),
            current_sources_published=False,
            original_index_unchanged=True,
        )
    )
)
