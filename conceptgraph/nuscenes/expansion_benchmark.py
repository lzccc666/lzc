"""Manifest preparation with source provenance and strict log isolation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def iter_records(path: Path, chunk_size: int = 1024 * 1024):
    """Read a JSON array without materializing multi-gigabyte nuScenes tables."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, cursor, eof = "", 0, False

        def refill():
            nonlocal buffer, cursor, eof
            more = handle.read(chunk_size)
            buffer = buffer[cursor:] + more
            cursor = 0
            eof = not more

        def whitespace():
            nonlocal cursor
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer) or eof:
                    return
                refill()

        refill()
        whitespace()
        if cursor == len(buffer) or buffer[cursor] != "[":
            raise ValueError(f"not a JSON array: {path}")
        cursor += 1
        first = True
        while True:
            whitespace()
            if cursor < len(buffer) and buffer[cursor] == "]":
                cursor += 1
                break
            if not first:
                if cursor == len(buffer) or buffer[cursor] != ",":
                    raise ValueError(f"missing array delimiter: {path}")
                cursor += 1
                whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(value, dict):
                raise ValueError(f"expected object record: {path}")
            cursor = end
            first = False
            yield value
        whitespace()
        if cursor != len(buffer):
            raise ValueError(f"trailing JSON content: {path}")


def content_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def scene_strata(description: str, location: str) -> dict:
    text = description.lower()
    return {
        "location": location,
        "road_context": "parking" if "park" in text else "road",
        "intersection": any(word in text for word in ("intersection", "junction")),
        "turn": any(word in text for word in ("turn", "bend", "curve")),
        "light": "night" if "night" in text else "day_or_unspecified",
        "weather": "rain" if "rain" in text else "dry_or_unspecified",
        "source": "official scene description; unspecified is not verified clear weather",
    }


def select_scenes(rows: list[dict], legacy: dict, target: int = 50, test_count: int = 10):
    """Select by source completeness/description only, before method outcomes exist."""
    if target < len(legacy) + test_count or test_count < 1:
        raise ValueError("insufficient target size for legacy and held-out scenes")
    by_name = {row["scene"]: row for row in rows}
    if len(by_name) != len(rows) or set(legacy) - set(by_name):
        raise ValueError("duplicate scenes or legacy metadata missing")
    legacy_logs = {by_name[name]["log_token"] for name in legacy}
    available = [r for r in rows if r["raw_complete"] or r.get("archive_complete")]
    candidates = [r for r in available if r["scene"] not in legacy]
    # Interleave strata instead of maximizing the old parking-success score.
    buckets = defaultdict(list)
    for row in candidates:
        strata = row["strata"]
        key = tuple(strata[k] for k in ("location", "road_context", "light", "weather"))
        buckets[key].append(row)
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    heldout = []
    for row in ordered:
        if row["log_token"] not in legacy_logs and len(heldout) < test_count:
            heldout.append(row)
    test_logs = {r["log_token"] for r in heldout}
    development = [
        r for r in ordered if r["log_token"] not in test_logs
    ][: max(0, target - len(legacy) - len(heldout))]
    chosen = []
    for row, role in (
        [(by_name[name], "legacy_regression") for name in sorted(legacy)]
        + [(r, "sealed_test") for r in heldout]
        + [(r, "development") for r in development]
    ):
        chosen.append({**row, "role": role, "original_split": legacy.get(row["scene"])})
    split = {r["scene"]: r["role"] for r in chosen}
    audit = {
        "same_log_leakage": sorted(
            test_logs & {r["log_token"] for r in chosen if r["role"] != "sealed_test"}
        ),
        "legacy_test_overlap": sorted(set(legacy) & {r["scene"] for r in heldout}),
        "selected_count": len(chosen),
        "test_count": len(heldout),
        "target_count": target,
        "minimum_scene_count_met": len(chosen) >= 30,
        "minimum_test_count_met": len(heldout) >= test_count,
        "selection_uses_method_results": False,
        "near_route_and_crop_audit": "pending; log isolation alone does not establish geographic independence",
        "test_labels_opened": False,
    }
    return {"scenes": chosen, "scene_to_split": split, "audit": audit, "sha256": content_sha256(chosen)}


def inventory(metadata: Path, data_roots: list[Path], converted_root: Path, archive_members: set[str]):
    scenes = list(iter_records(metadata / "scene.json"))
    logs = {r["token"]: r for r in iter_records(metadata / "log.json")}
    samples = {r["token"]: r for r in iter_records(metadata / "sample.json")}
    channels = defaultdict(dict)
    for record in iter_records(metadata / "sample_data.json"):
        parts = Path(record["filename"]).parts
        if record["is_key_frame"] and len(parts) == 3 and parts[0] == "samples" and parts[1] in {"CAM_FRONT", "LIDAR_TOP"}:
            sample_channels = channels[record["sample_token"]]
            if parts[1] in sample_channels:
                raise ValueError("duplicate keyframe sensor channel")
            sample_channels[parts[1]] = record
    result, file_index = [], {}
    for scene in sorted(scenes, key=lambda r: r["name"]):
        names, sequence = [], []
        token = scene["first_sample_token"]
        visited = set()
        while token:
            if token in visited or token not in samples:
                raise ValueError("broken sample chain")
            visited.add(token)
            sample = samples[token]
            if sample["scene_token"] != scene["token"]:
                raise ValueError("cross-scene sample chain")
            sensor_records = channels.get(token, {})
            for channel in ("CAM_FRONT", "LIDAR_TOP"):
                if channel in sensor_records:
                    names.append(sensor_records[channel]["filename"])
            sequence.append({"sample_token": token, "timestamp": sample["timestamp"], "sensors": sensor_records})
            token = sample["next"]
        found = {name: next((str(root / name) for root in data_roots if (root / name).is_file()), None) for name in names}
        expected = int(scene["nbr_samples"]) * 2
        directory = converted_root / scene["name"]
        counts = {k: len(list((directory / k).glob("*"))) for k in ("color", "depth")}
        pose_count = 0
        if (directory / "poses.txt").exists():
            pose_count = len((directory / "poses.txt").read_text().splitlines())
        counts["pose_rows"] = pose_count
        counts["detections"] = len(list((directory / "exp_streamlined_yolo_outdoor_objects_v1/detections").glob("*.pkl.gz")))
        counts["crops"] = len(list((directory / "exp_qwen2vl_semantics_v1").rglob("*.jpg")))
        counts["scene_graph"] = (directory / "exp_qwen2vl_semantics_v1/scene_graph.json").is_file()
        result.append({
            "scene": scene["name"], "scene_token": scene["token"], "log_token": scene["log_token"],
            "description": scene["description"], "frame_count": scene["nbr_samples"],
            "strata": scene_strata(scene["description"], logs[scene["log_token"]]["location"]),
            "raw_expected": expected, "raw_found": sum(p is not None for p in found.values()),
            "raw_complete": len(names) == expected and all(found.values()),
            "archive_complete": len(names) == expected and all(found[n] or n in archive_members for n in names),
            "converted_counts": counts,
            "converted_complete": counts["color"] == counts["depth"] == pose_count == scene["nbr_samples"],
        })
        file_index[scene["name"]] = {"sequence": sequence, "source_files": found}
    return result, file_index


def paired_scene_bootstrap(rows: list[dict], method_a: str, method_b: str, metric: str, *, seed: int = 17, iterations: int = 2000):
    """Pair queries first, then resample scenes; retain missing-pair diagnostics."""
    import numpy as np

    if iterations < 2000:
        raise ValueError("at least 2000 bootstrap samples required")
    indexed = {}
    for row in rows:
        key = (row["scene"], row["query_id"], row["method"])
        if key in indexed:
            raise ValueError("duplicate query/method; select a declared repeat first")
        indexed[key] = row.get(metric)
    queries = sorted({(r["scene"], r["query_id"]) for r in rows})
    by_scene = defaultdict(list)
    missing = []
    for scene, query in queries:
        a, b = (indexed.get((scene, query, m)) for m in (method_a, method_b))
        if a is None or b is None:
            missing.append([scene, query])
        elif not np.isfinite([a, b]).all():
            raise ValueError("non-finite metric")
        else:
            by_scene[scene].append(float(b) - float(a))
    means = np.asarray([np.mean(values) for _, values in sorted(by_scene.items())])
    report = {"metric": metric, "difference": f"{method_b} - {method_a}", "scene_count": len(means), "paired_queries": sum(map(len, by_scene.values())), "missing_pairs": missing, "seed": seed, "iterations": iterations}
    if len(means) < 2:
        return {**report, "mean_difference": None if not len(means) else float(means[0]), "ci95": None, "status": "insufficient_independent_scenes"}
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(means, (iterations, len(means))), axis=1)
    return {**report, "mean_difference": float(means.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(), "status": "computed", "resampling_unit": "scene", "scene_macro_average": True}


def summarize_inventory(rows):
    return {"metadata_scenes": len(rows), "raw_complete_scenes": sum(r["raw_complete"] for r in rows), "archive_or_raw_complete_scenes": sum(r["archive_complete"] for r in rows), "converted_complete_scenes": sum(r["converted_complete"] for r in rows), "locations": dict(Counter(r["strata"]["location"] for r in rows))}
