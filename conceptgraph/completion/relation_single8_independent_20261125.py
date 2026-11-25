"""Independent angular/pair checks and old-node/new-edge pilot comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

from conceptgraph.scenegraph import relation_single8_20261125 as tested

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/results/nuscenes/validation/iterations"
REPORT = ROOT / "outputs/reports/local_scope_audit_20261125"
NAMES = ("front", "front_left", "left", "back_left", "back", "back_right", "right", "front_right")
ADDED = {"relation_contract", "relation_statistics", "relation_provider_sha256"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def independent_label(delta, forward):
    """Nearest of eight unit rays, without the tested atan2/bin implementation."""
    fx, fy = map(float, forward)
    length = math.hypot(fx, fy)
    fx, fy = fx / length, fy / length
    front = delta[0] * fx + delta[1] * fy
    left = -delta[0] * fy + delta[1] * fx
    scores = [front * math.cos(k * math.pi / 4) + left * math.sin(k * math.pi / 4) for k in range(8)]
    maximum = max(scores)
    tie = [i for i, value in enumerate(scores) if maximum - value <= 8 * math.ulp(max(abs(maximum), 1.0))]
    index = 0 if 0 in tie and 7 in tie else max(tie)
    return NAMES[index]


def fixture_checks():
    checks = 0
    for yaw in (0.0, .31, math.pi / 2, -1.7):
        f = [math.cos(yaw), math.sin(yaw)]
        left = [-f[1], f[0]]
        for k, label in enumerate(NAMES):
            angle = math.radians(k * 45)
            delta = [5 * (f[j] * math.cos(angle) + left[j] * math.sin(angle)) for j in range(2)]
            assert tested.direction_for_delta(delta, f) == independent_label(delta, f) == label
            assert tested.direction_for_delta([-v for v in delta], f) == NAMES[(k + 4) % 8]
            checks += 2
        for k in range(8):
            for perturbation in (-1e-6, 0.0, 1e-6):
                angle = math.radians(22.5 + 45 * k + perturbation)
                delta = [5 * (f[j] * math.cos(angle) + left[j] * math.sin(angle)) for j in range(2)]
                expected = NAMES[(k + int(perturbation >= 0)) % 8]
                assert tested.direction_for_delta(delta, f) == expected
                assert independent_label(delta, f) == expected
                checks += 1
    original = [dict(id="a", center=[-3., 4.], motion_state="unknown", confidence=0),
                dict(id="b", center=[0., 0.], motion_state="dynamic", confidence=0)]
    before = copy.deepcopy(original)
    snapshot = tested.build_edge_snapshot(original, [0., 1.])
    assert original == before and len(snapshot["edges"]) == 1
    edge = snapshot["edges"][0]
    assert edge == dict(source="a", target="b", relation="front_left", distance_m=5.)
    reverse = tested.query_pair(snapshot["edges"], "b", "a")
    assert reverse == dict(source="b", target="a", relation="back_right", distance_m=5.)
    assert len(snapshot["edges"]) == 1
    assert tested.canonical_fact("z", edge["relation"], "y") == ("y", "back_right", "z")
    assert independent_label([3., -4.], [0., 1.]) == "back_right"
    lexical = tested.build_edges_from_centers([dict(id="2", center=[-3., 4.]), dict(id="10", center=[0., 0.])], [0., 1.])
    assert lexical[0]["source"] == "10" and lexical[0]["relation"] == "back_right"
    for distance, count in ((0., 0), (1e-6, 0), (1e-6 + 1e-10, 1), (.2, 1), (6., 1), (12., 1), (12.000001, 0)):
        result = tested.build_edge_snapshot([dict(id="a", center=[0., distance]), dict(id="b", center=[0., 0.])], [0., 1.])
        assert len(result["edges"]) == count
        if distance <= 1e-6:
            assert result["statistics"]["coincident_pairs"] == 1
        checks += 1
    invalid = tested.build_edge_snapshot([dict(id="a", center=[float("nan"), 0.]), dict(id="b", center=[0., 0.])], [0., 1.])
    assert invalid["statistics"]["invalid_nodes"] == 1 and not invalid["edges"]
    return dict(passed=True, analytical_direction_boundary_rotated_checks=checks,
        single_canonical_pair_and_inverse_query=True, no_confidence_or_motion_pruning=True,
        source_relative_to_target=True, stable_ID_lexical_order=True,
        mapped_ID_order_reversal_fixture=dict(predicted_edge=["a", "front_left", "b"],
            mapping={"a": "z", "b": "y"}, expected_canonical_GT_fact=["y", "back_right", "z"]),
        actual_GT_evaluator_fixture_checked=False)


def audit_payload(old, new, forward):
    stripped = {k: v for k, v in new.items() if k not in ADDED}
    assert {k: v for k, v in old.items() if k != "edges"} == {k: v for k, v in stripped.items() if k != "edges"}
    assert new["relation_provider_sha256"] == sha(tested.__file__)
    predictions = {str(p["node_id"]): p for p in new["predictions"]}
    assert len(predictions) == len(new["predictions"])
    registry = {str(k): str(v) for k, v in new["relation_node_registry"].items()}
    # The unchanged upstream registry also retains historical stable IDs.
    # Only currently exported predictions produce this frame's relations.
    assert set(predictions) <= set(registry.values())
    assert len(registry.values()) == len(set(registry.values()))
    expected = {}
    coincident = far = 0
    for first, second in itertools.combinations(sorted(predictions), 2):
        a, b = predictions[first]["center"], predictions[second]["center"]
        delta = [a[0] - b[0], a[1] - b[1]]
        distance = math.hypot(*delta)
        if distance <= 1e-6:
            coincident += 1
        elif distance > 12:
            far += 1
        else:
            expected[first, second] = (independent_label(delta, forward), distance)
    actual = {}
    maximum_distance_error = 0.0
    for edge in new["edges"]:
        source, target = registry[str(edge["source"])], registry[str(edge["target"])]
        assert source < target and (source, target) not in actual
        assert edge["relation"] in NAMES and set(edge) == {"source", "target", "relation", "distance_m"}
        label, distance = expected[source, target]
        assert edge["relation"] == label
        error = abs(edge["distance_m"] - distance)
        assert error < 1e-10
        maximum_distance_error = max(maximum_distance_error, error)
        actual[source, target] = edge["relation"]
    assert set(actual) == set(expected)
    stats = new["relation_statistics"]
    assert stats["stored_edges"] == len(expected) and stats["coincident_pairs"] == coincident and stats["outside_distance_pairs"] == far
    return len(actual), coincident, maximum_distance_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    source_sha = sha(tested.__file__)
    result = dict(core_fixture=fixture_checks(), source_sha256=source_sha,
        independent_geometry_method="nearest8 unit-ray dot products, separate from tested atan2 sectors")
    if args.pilot:
        old_root = RUNS / "nuscenes_full_20261125_resource_v1"
        new_root = RUNS / "nuscenes_full_20261125_single8_pilot_v1"
        totals, scenes = Counter(), []
        max_error = 0.0
        for number in range(1, 9):
            name = f"scene-{number:04d}"
            old, new = old_root / name, new_root / name
            complete = json.loads((new / "complete.json").read_text())
            auxiliary = json.loads((new / "sharedscan_complete.json").read_text())
            assert complete["complete"] and auxiliary["complete"]
            assert sha(new / "frames.jsonl.gz") == complete["frames_sha256"]
            assert sha(new / "sharedscan_frames.jsonl.gz") == auxiliary["frames_sha256"]
            old_frames, new_frames = list(records(old / "frames.jsonl.gz")), list(records(new / "frames.jsonl.gz"))
            assert len(old_frames) == len(new_frames) == complete["counts"]["frames"]
            forwards = {}
            for before, after in zip(old_frames, new_frames):
                assert before["sample_token"] == after["sample_token"]
                assert {k: v for k, v in before.items() if k != "methods"} == {k: v for k, v in after.items() if k not in {"methods", *ADDED}}
                forward = [before["views"]["CAM_FRONT"]["camera_pose"][i][2] for i in (0, 1)]
                forwards[before["sample_token"]] = forward
                assert before["methods"].keys() == after["methods"].keys()
                for method in before["methods"]:
                    edges, coincident, error = audit_payload(before["methods"][method], after["methods"][method], forward)
                    totals[method + ":frames"] += 1
                    totals[method + ":edges"] += edges
                    totals[method + ":coincident"] += coincident
                    max_error = max(max_error, error)
            before_aux, after_aux = list(records(old / "sharedscan_frames.jsonl.gz")), list(records(new / "sharedscan_frames.jsonl.gz"))
            assert len(before_aux) == len(after_aux) == len(old_frames)
            for before, after in zip(before_aux, after_aux):
                assert {k: v for k, v in before.items() if k != "payload"} == {k: v for k, v in after.items() if k not in {"payload", *ADDED}}
                edges, coincident, error = audit_payload(before["payload"], after["payload"], forwards[before["sample_token"]])
                totals["S2_6_sharedscan:frames"] += 1
                totals["S2_6_sharedscan:edges"] += edges
                totals["S2_6_sharedscan:coincident"] += coincident
                max_error = max(max_error, error)
            scenes.append(dict(scene=name, frames=len(old_frames), complete_sha256=sha(new / "complete.json")))
        assert sum(count for key, count in totals.items() if key.endswith(":frames")) == 2226
        result.update(pilot_passed=True, scenes=scenes, totals=dict(totals),
            maximum_distance_error_m=max_error, all_nonedge_fields_exact=True,
            method_frames=2226, new_GPU_calls=0, GT_not_read=True)
    assert sha(tested.__file__) == source_sha
    result.update(audit_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - started,
        audit_source_sha256=sha(__file__))
    target = REPORT / ("single8_pilot_independent_v1.json" if args.pilot else "single8_core_independent_v1.json")
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}))


if __name__ == "__main__":
    main()
