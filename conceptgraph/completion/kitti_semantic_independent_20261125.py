"""Audit all locally paired SemanticKITTI records without reading sensor payloads."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/reports/kitti_local_all_20261125"
DEST = ROOT / "outputs/reports/local_scope_audit_20261125/semantic_pairs_independent_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    current_path = REPORT / "semantic_RGB_LiDAR_pose_pair_inventory_v3.json"
    previous_path = REPORT / "semantic_RGB_LiDAR_pose_pair_inventory_v2.json"
    current = json.loads(current_path.read_text())
    previous = json.loads(previous_path.read_text())
    mapping_path = REPORT / "official_odometry_raw_mapping_receipt_v1.json"
    mapping = json.loads(mapping_path.read_text())
    headers_path = REPORT / "depth_tar_header_inventory_v1.json"
    headers = json.loads(headers_path.read_text())
    amendment_path = REPORT / "semantic_resume_amendment_v3.json"
    amendment = json.loads(amendment_path.read_text())
    progress = json.loads((REPORT / "semantic_resume_progress_v3.json").read_text())
    scope = json.loads((REPORT / "SemanticKITTI_22seq_applicability_v1.json").read_text())
    assert current["complete"] and not current["bounded_partial"]
    assert sha(current_path) == "ed51e4cf4675d191bf3535b95b8c6f5cb0a5316d50bf929aa7c67df238466805"
    assert current["official_mapping_metadata_source"] == mapping
    assert amendment["mapping_sha256"] == sha(mapping_path)
    assert amendment["prior_receipt_sha256"] == sha(previous_path)
    assert amendment["full_directory_inventory_sha256"] == sha(headers_path)
    assert current["resume_amendment_sha256"] == sha(amendment_path)
    for path, digest in amendment["source_sha256"].items():
        assert sha(path) == digest
    expected = {
        "00": ("2011_10_03_drive_0027", 0, 4540),
        "01": ("2011_10_03_drive_0042", 0, 1100),
        "02": ("2011_10_03_drive_0034", 0, 4660),
        "03": ("2011_09_26_drive_0067", 0, 800),
        "04": ("2011_09_30_drive_0016", 0, 270),
        "05": ("2011_09_30_drive_0018", 0, 2760),
        "06": ("2011_09_30_drive_0020", 0, 1100),
        "07": ("2011_09_30_drive_0027", 0, 1100),
        "08": ("2011_09_30_drive_0028", 1100, 5170),
        "09": ("2011_09_30_drive_0033", 0, 1590),
        "10": ("2011_09_30_drive_0034", 0, 1200),
    }
    assert {r["sequence"]: (r["raw_drive"], r["raw_start"], r["raw_end"]) for r in mapping["rows"]} == expected
    pairs = {(r["sequence"], r["frame_idx"]): r for r in current["pairs"]}
    assert len(pairs) == len(current["pairs"]) == current["total_common_frames"] == 22400
    assert all(pairs[r["sequence"], r["frame_idx"]] == r for r in previous["pairs"])
    assert current["reused_prior_pairs"] == len(previous["pairs"]) == 10883
    assert amendment["skipped_certified_sequences"] == ["00", "01", "02"]
    assert current["new_headers_read"] == progress["new_headers_read"] == 12665
    matrix = {r["sequence"]: r for r in current["matrix"]}
    assert set(matrix) == set(expected)
    archive = Path(headers["archive"])
    stat = archive.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (headers["archive_bytes"], headers["archive_mtime_ns"])
    semantic = Path("/root/autodl-pub/SemanticKITTI")
    counts = Counter()
    sampled_headers = []
    metadata_shas = {}
    with zipfile.ZipFile(semantic / "data_odometry_velodyne.zip") as lidar, \
            zipfile.ZipFile(semantic / "data_odometry_labels.zip") as labels, \
            zipfile.ZipFile(semantic / "data_odometry_calib.zip") as calib:
        lidar_info = {r.filename: r for r in lidar.infolist()}
        label_info = {r.filename: r for r in labels.infolist()}
        for sequence, (drive, first, last) in expected.items():
            length = last - first + 1
            row = matrix[sequence]
            prefix = f"dataset/sequences/{sequence}/"
            pose_bytes = labels.read(prefix + "poses.txt")
            time_bytes = calib.read(prefix + "times.txt")
            calibration = calib.read(prefix + "calib.txt")
            poses = np.loadtxt(pose_bytes.decode().splitlines()).reshape(-1, 3, 4)
            times = np.atleast_1d(np.loadtxt(time_bytes.decode().splitlines()))
            assert len(poses) == len(times) == length
            assert np.isfinite(poses).all() and np.isfinite(times).all() and (np.diff(times) > 0).all()
            for key, content in [("pose_sha256", pose_bytes), ("times_sha256", time_bytes), ("calib_sha256", calibration)]:
                assert hashlib.sha256(content).hexdigest() == row[key]
            metadata_shas[sequence] = {key: row[key] for key in ("pose_sha256", "times_sha256", "calib_sha256")}
            actual = {frame for seq, frame in pairs if seq == sequence}
            if sequence == "03":
                assert not actual and not row["common_complete"]
                assert headers["drives"].get(drive + "_sync", {}).get("RGB2", 0) == 0
                continue
            assert actual == set(range(length)) == set(row["common_frame_ids"])
            assert row["common_complete"] and row["common_RGB_LiDAR_SuMa_pose_time_frames"] == length
            for frame in sorted(actual):
                pair = pairs[sequence, frame]
                rgb = pair["RGB"]
                raw = first + frame
                assert pair["raw_drive"] == drive and rgb["raw_frame"] == raw
                assert rgb["name"] == f"KITTI_Depth_Completion/raw/{drive[:10]}/{drive}_sync/image_02/data/{raw:010d}.png"
                assert rgb["bytes"] > 0 and rgb["offset_data"] % 512 == 0
                assert 512 <= rgb["offset_data"] < rgb["offset_data"] + rgb["bytes"] <= stat.st_size
                member = prefix + f"velodyne/{frame:06d}.bin"
                li = lidar_info[member]
                assert (pair["LiDAR_member"], pair["LiDAR_CRC32"], pair["LiDAR_bytes"]) == (member, li.CRC, li.file_size)
                assert li.file_size % 16 == 0
                label = label_info[prefix + f"labels/{frame:06d}.label"]
                assert label.file_size == li.file_size // 4
                assert pair["point_instance_label_available"] is True
                assert pair["pose_row"] == pair["times_row"] == frame
                counts[sequence] += 1
            # One real 512-byte tar header per available sequence; no image/LiDAR payload read.
            pair = pairs[sequence, length // 2]
            with archive.open("rb") as source:
                source.seek(pair["RGB"]["offset_data"] - 512)
                header = tarfile.TarInfo.frombuf(source.read(512), "utf-8", "surrogateescape")
            assert header.isfile() and header.name == pair["RGB"]["name"] and header.size == pair["RGB"]["bytes"]
            sampled_headers.append(dict(sequence=sequence, frame_idx=pair["frame_idx"], name=header.name))
    assert sum(counts.values()) == 22400 and len(counts) == current["matched_sequences"] == 10
    assert set(counts) == set(current["completed_RGB_sequences"])
    assert sum(m["LiDAR_frames"] for m in scope["matrix"]) == 43552
    assert sum(m["point_label_frames"] for m in scope["matrix"]) == 23201
    assert len(scope["matrix"]) == 22 and all(not m["object_3D_box_GT"] for m in scope["matrix"])
    result = dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        inventory_sha256=sha(current_path), source_sha256=sha(__file__),
        pair_count=22400, matched_sequences=10, by_sequence=dict(counts),
        retained_missing_sequence="03:801 LiDAR/point-label/pose/time frames lack matching local raw RGB",
        sequence08_raw_offset=1100, all_prior_pairs_byte_equivalent=True,
        reused_pairs=10883, no_repeat_of_certified_00_01_02_headers_in_resume_source=True,
        current_tar_header_samples=sampled_headers, sensor_payload_bytes_read=0,
        independent_large_archive_CRC_or_SHA=False,
        original_mapping_metadata_download_SHA_repeated=False,
        supplied_pose_time_calibration_SHA=metadata_shas,
        pose_scope="provider SuMa SLAM, not raw OXTS; strict causal pose not certified",
        metric_scope="data readiness only; point semantic/instance labels are not Object3D box centers",
        current_Object_Tracking_results_changed=False, new_GPU_calls=0,
        elapsed_s=time.monotonic() - started)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("passed", "pair_count", "matched_sequences", "by_sequence", "elapsed_s")}))


if __name__ == "__main__":
    main()
