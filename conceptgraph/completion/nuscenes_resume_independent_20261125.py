"""Authenticate preserved full-CRC/export evidence behind resumed member scans."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/processed/nuscenes_full_20261125_v1"
REPORT = ROOT / "outputs/reports/local_scope_audit_20261125"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=int, required=True, choices=(3, 4, 5))
    number = parser.parse_args().archive
    started = time.monotonic()
    name = f"archive{number:02d}"
    closed_path = DATA / "archives" / (name + "_complete.json")
    closed = json.loads(closed_path.read_text())
    assert closed["complete"] and closed["technical_revision"] == "resume_exported_crc_index_v3"
    assert closed["new_CRC_validation"] is False and closed["imported_index_CRC_disabled"] is True
    prereg_path = Path(closed["CRC_validation_provenance"])
    prereg = json.loads(prereg_path.read_text())
    assert prereg["archive"] == number and prereg["new_CRC_pass"] is False
    old_source = ROOT / "conceptgraph/completion/nuscenes_full_20261125_index_v2.py"
    revision = json.loads((DATA / "index_technical_revision_v2.json").read_text())
    for filename, expected in revision["source_sha256"].items():
        assert sha(filename) == expected
    assert sha(old_source) == prereg["original_index_source_sha256"]
    old_text = old_source.read_text()
    assert "skip_crc_check=False" in old_text
    export_line = next(i for i, text in enumerate(old_text.splitlines(), 1)
                       if "source.build_full_index();source.export_index" in text)
    old_log = Path(prereg["predecessor_log"])
    assert sha(old_log) == prereg["predecessor_log_sha256"]
    log = old_log.read_text()
    assert "TimeoutError: phaseA deadline" in log
    assert (f"line {export_line}, in index" in log and "base.guard(deadline)" in log) or '"stage": "seek_index_complete"' in log
    assert sha(prereg["reused_index"]) == prereg["reused_index_sha256"] == closed["seek_index_sha256"]
    assert prereg["reused_index"] == closed["seek_index"]
    for filename, expected in prereg["source_sha256"].items():
        assert sha(filename) == expected
    stat = Path(closed["archive"]).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (closed["archive_bytes"], closed["archive_mtime_ns"])
    assert prereg["source_archive_stat"] == dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    assert prereg["deadline_unix"] == 1795617120.0
    assert datetime.fromisoformat(prereg["created_utc"]) < datetime.fromisoformat(closed["ended_utc"])
    member_audit_path = REPORT / (f"nuscenes_archive{number:02d}_independent_v1.json")
    if not member_audit_path.exists():
        from conceptgraph.completion.local_scope_audit_20261125 import nuscenes_archive
        nuscenes_archive(number)
    member_audit = json.loads(member_audit_path.read_text())
    assert member_audit["passed"] and member_audit["complete_receipt_sha256"] == sha(closed_path)
    result = dict(passed=True, archive=number, audit_utc=datetime.now(timezone.utc).isoformat(),
        complete_sha256=sha(closed_path), preregistration_sha256=sha(prereg_path),
        preserved_predecessor_trace_sha256=sha(old_log), original_full_CRC_export_control_flow_authenticated=True,
        imported_index_did_not_rerun_CRC=True, full_source_SHA_not_independently_repeated=True,
        closed_member_and_seek_index_audit_sha256=sha(member_audit_path),
        original_failure_preserved=True, new_GPU_calls=0,
        elapsed_s=time.monotonic() - started, audit_source_sha256=sha(__file__))
    target = REPORT / (f"nuscenes_archive{number:02d}_resume_chain_independent_v1.json")
    with target.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
