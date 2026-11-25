"""Check repeated scenes while excluding exactly five mapper wall durations."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/results/nuscenes/validation/iterations"
PILOT = RUNS / "nuscenes_full_20261125_resource_v1"
REPLAY = RUNS / "nuscenes_full_20261125_cpu_runtime_v6_replay_v1"
METHODS = ("original_fusion", "R1", "S2_1", "S2_3", "S2_6")
DEST = ROOT / "outputs/reports/local_scope_audit_20261125/nuscenes_cpu_runtime_v6_independent_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(path):
    h = hashlib.sha256()
    with gzip.open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def normalized(frame):
    # No other duration, timestamp, capture time, source, or metadata is removed.
    result = copy.deepcopy(frame)
    durations = result["mapper_seconds"]
    assert set(durations) == set(METHODS)
    for method in METHODS:
        value = durations[method]
        assert isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        durations[method] = "independent_audit_excludes_only_this_wall_duration"
    return result


def main():
    started = time.monotonic()
    prereg_path = REPLAY / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    assert prereg["frozen_sequence_same_worker"] == [4, 5, 4, 5]
    for filename, expected in prereg["source_sha256"].items():
        assert sha(filename) == expected
    checks = []
    comparisons = []
    for number in (4, 5):
        scene = f"scene-{number:04d}"
        old, new = PILOT / scene, REPLAY / scene
        old_check_path = old / "attempts/cpu_v6/runtime_v6_check.json"
        new_check_path = new / "attempts/cpu_v6/runtime_v6_check.json"
        first = json.loads(old_check_path.read_text())
        repeat = json.loads(new_check_path.read_text())
        assert first["scene"] == repeat["scene"] == scene
        assert first["repeat"] is False and repeat["repeat"] is True
        checks.extend((first, repeat))
        old_path, new_path = old / "frames.jsonl.gz", new / "frames.jsonl.gz"
        with gzip.open(old_path, "rt") as source:
            old_frames = [json.loads(line) for line in source]
        with gzip.open(new_path, "rt") as source:
            new_frames = [json.loads(line) for line in source]
        assert len(old_frames) == len(new_frames) == first["frames"] == repeat["frames"]
        difference_counts = Counter()
        for left, right in zip(old_frames, new_frames):
            assert set(left["methods"]) == set(right["methods"]) == {"single_frame", *METHODS}
            assert normalized(left) == normalized(right), "non-wall-duration payload changed"
            for method in METHODS:
                if left["mapper_seconds"][method] != right["mapper_seconds"][method]:
                    difference_counts[method] += 1
        assert content_sha(old_path) == first["main_content_sha256"]
        assert content_sha(new_path) == repeat["main_content_sha256"]
        auxiliary = content_sha(old / "sharedscan_frames.jsonl.gz")
        assert auxiliary == content_sha(new / "sharedscan_frames.jsonl.gz")
        assert auxiliary == first["auxiliary_content_sha256"] == repeat["auxiliary_content_sha256"]
        comparisons.append(dict(scene=scene, frames=len(old_frames),
            main_payload_equal_except_exact_mapper_wall_duration_paths=True,
            differences_by_allowed_method=dict(difference_counts),
            auxiliary_decompressed_bytes_SHA_equal=True,
            original_main_gzip_sha256=sha(old_path), repeated_main_gzip_sha256=sha(new_path)))
    assert len({r["PID"] for r in checks}) == 1
    assert all(r["interop_threads"] == 1 and r["CUDA_initialized"] is False and r["all7closed"] for r in checks)
    for number in (4, 5, 6):
        folder = PILOT / f"scene-{number:04d}"
        main_closed = json.loads((folder / "complete.json").read_text())
        auxiliary_closed = json.loads((folder / "sharedscan_complete.json").read_text())
        assert main_closed["complete"] and auxiliary_closed["complete"]
        assert sha(folder / "frames.jsonl.gz") == main_closed["frames_sha256"]
    data = ROOT / "dataset/processed/nuscenes_full_20261125_v1"
    protected = json.loads((data / "phaseB_source_preseal_v1.json").read_text())
    # Receipts preserve the frozen implementation; this audit changes no mapper source.
    frozen = protected.get("source_sha256", protected.get("sources_sha256"))
    if frozen is None:
        frozen = protected["sources"]
    assert isinstance(frozen, dict)
    for filename, expected in frozen.items():
        assert sha(filename) == expected
    result = dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        same_worker_PID=checks[0]["PID"], frozen_sequence=[4, 5, 4, 5], comparisons=comparisons,
        allowed_exclusion_paths=[["mapper_seconds", method] for method in METHODS],
        all_other_fields_compared_exact=True, all_three_cached_scenes_closed=True,
        source_preregistration_sha256=sha(prereg_path), original_overstrict_SHA_failure_preserved=True,
        new_GPU_calls=0, scope="runtime worker reuse and repeated original CPU mapper outputs; no new quality score",
        elapsed_s=time.monotonic() - started, audit_source_sha256=sha(__file__))
    with DEST.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
