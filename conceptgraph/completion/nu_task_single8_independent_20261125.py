"""Recompute directional answers with independent geometry and literal parsing."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from conceptgraph.completion.relation_single8_independent_20261125 import independent_label

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "dataset/processed/nu_task_replay_20261125_v1"
DATA = ROOT / "dataset/processed/nu_task_single8_20261125_v1"
REPORT = ROOT / "outputs/reports/local_scope_audit_20261125"
CATEGORY = {"小汽车": "car", "卡车": "truck", "公交车": "bus", "行人": "pedestrian", "自行车": "bicycle", "摩托车": "motorcycle"}
PHRASES = (("左前方", "front_left"), ("右前方", "front_right"), ("左后方", "back_left"), ("右后方", "back_right"),
           ("前方", "front"), ("后方", "back"), ("左侧", "left"), ("右侧", "right"))
INCLUDES = dict(front={"front", "front_left", "front_right"}, back={"back", "back_left", "back_right"},
                left={"left", "front_left", "back_left"}, right={"right", "front_right", "back_right"},
                front_left={"front_left"}, front_right={"front_right"}, back_left={"back_left"}, back_right={"back_right"})


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for section in ("source_sha256", "frozen_inputs_sha256"):
        for path, expected in manifest[section].items():
            assert sha(path) == expected
    assert sha(DATA / "method_tasks.jsonl") == sha(OLD / "method_tasks.jsonl")
    tasks = rows(DATA / "method_tasks.jsonl")
    previous = {r["task_id"]: r for r in rows(OLD / "evaluation_only/task_labels.jsonl")}
    current = {r["task_id"]: r for r in rows(DATA / "evaluation_only/task_labels.jsonl")}
    ledger = {r["task_id"]: r for r in rows(DATA / "evaluation_only/label_change_ledger.jsonl")}
    old_truth = {(r["protocol"], r["scene"], r["original_frame_idx"]): r for r in rows(OLD / "evaluation_only/frame_truth.jsonl")}
    context = {(r["protocol"], r["scene"], r["original_frame_idx"]): r for r in rows(DATA / "evaluation_only/GT_context.jsonl")}
    assert len(tasks) == len(previous) == len(current) == len(ledger) == 2315
    assert len(context) == len(old_truth) == 502 and context.keys() == old_truth.keys()
    for key, value in context.items():
        old_objects = old_truth[key]["objects"]
        if key[0] == "legacy1115":
            assert [{k: v for k, v in r.items() if k != "evaluation_motion"} for r in value["objects"]] == old_objects
        else:
            assert value["objects"] == old_objects
        assert value["sample_token"] == old_truth[key]["sample_token"]
    stats = defaultdict(Counter)
    count = Counter()
    for task in tasks:
        identity = task["task_id"]
        before, after, entry = previous[identity], current[identity], ledger[identity]
        relation = next(((phrase, name) for phrase, name in PHRASES if phrase in task["query_text"]), None)
        assert bool(relation) == entry["directional"]
        count["directional" if relation else "class_nearest"] += 1
        expected = copy.deepcopy(before)
        if before["label_status"] != "scorable":
            count["pending_exact"] += 1
        elif relation:
            phrase, requested = relation
            prefix, suffix = task["query_text"].split(phrase, 1)
            anchors = {value for key, value in CATEGORY.items() if key in prefix}
            targets = {value for key, value in CATEGORY.items() if key in suffix}
            assert len(anchors) == len(targets) == 1
            ctx = context[task["protocol"], task["scene"], task["original_frame_idx"]]
            pose = ctx["camera_pose"]
            forward = [pose[i][2] for i in (0, 1)]
            references = [r for r in ctx["objects"] if r["category"] in anchors]
            if task["protocol"] == "unified1200":
                assert len(references) <= 1
            eligible = []
            for target in ctx["objects"]:
                if target["category"] not in targets or target["evaluation_motion"] != "static":
                    continue
                for reference in references:
                    if target["instance_token"] == reference["instance_token"]:
                        continue
                    delta = [target["center"][i] - reference["center"][i] for i in (0, 1)]
                    if 1e-6 < math.hypot(*delta) <= 12. and independent_label(delta, forward) in INCLUDES[requested]:
                        eligible.append(target)
                        break
            if task["protocol"] == "unified1200" and eligible:
                distance = {r["instance_token"]: math.hypot(r["center"][0] - pose[0][3], r["center"][1] - pose[1][3]) for r in eligible}
                minimum = min(distance.values())
                accepted = sorted(k for k, value in distance.items() if abs(value - minimum) <= 1e-9)
            else:
                accepted = sorted(r["instance_token"] for r in eligible)
            expected.update(positive=bool(accepted), acceptable_instances=accepted)
            if "reason" in expected:
                expected["reason"] = "single8_directional_predicate_recomputed_from_official_geometry"
        assert expected == after, identity
        assert entry["old_label"] == before and entry["new_label"] == after
        changed = before["acceptable_instances"] != after["acceptable_instances"]
        assert entry["acceptance_changed"] == changed
        count["changed"] += changed
        if changed:
            count["positive_to_negative"] += before["positive"] and not after["positive"]
            count["changed_still_positive"] += before["positive"] and after["positive"]
        protocol = stats[task["protocol"]]
        protocol["all"] += 1
        if after["label_status"] != "scorable":
            protocol["pending"] += 1
        else:
            protocol["positive" if after["positive"] else "negative"] += 1
    assert count["directional"] == 827 and count["class_nearest"] == 1488 and count["pending_exact"] == 60
    assert count["changed"] == manifest["label_changes"] == 64
    assert count["positive_to_negative"] == 41 and count["changed_still_positive"] == 23
    for protocol, values in manifest["task_counts"].items():
        assert dict(stats[protocol]) == {k: v for k, v in values.items() if v}
    result = dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        all_tasks=2315, official_context_frames=502, counts=dict(count),
        by_protocol={key: dict(value) for key, value in stats.items()},
        actual_shared_GT_builder_called=False, actual_navigation_matcher_called=False,
        independent_GEOMETRY="nearest8 ray dot products", independent_query_parse="literal category/phrase split; independent inclusion sets",
        all_original_class_nearest_and_pending_exact=True, all_GT_object_centers_IDs_and_visibility_scope_preserved=True,
        source_and_input_SHA_verified=True, manifest_sha256=sha(DATA / "manifest.json"),
        method_input_task_SHA_unchanged=True, predicted_maps_or_method_results_read=False,
        new_GPU_calls=0, elapsed_s=time.monotonic() - started, audit_source_sha256=sha(__file__))
    with (REPORT / "single8_task_GT_independent_v1.json").open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
