"""Attach evaluation-only provenance to frozen R1 false selections; no inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
R1 = PROJECT / "outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2"
RAW = PROJECT / "outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/queries.jsonl"
MOTION = PROJECT / "outputs/project_v2/thesis_completion_v1/annotation/official_motion_dev_001/evaluation_only_motion.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [(i, json.loads(line)) for i, line in enumerate(handle, 1)]


def key(row):
    return row["scene"], row["frame_idx"], row["query_index"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(RAW)
    old = {key(r): r for _, r in rows if r["method"] == "old_map_original_rule"}
    false = [(line, r) for line, r in rows if r["method"] == "R1_map_original_rule"
             and not r["positive"] and r["selected"]]
    frames = {}
    fingerprints = {str(p): sha(p) for p in (RAW, MOTION, Path(__file__))}
    for scene in sorted({r["scene"] for _, r in false}):
        path = R1 / scene / "frames.jsonl.gz"
        fingerprints[str(path)] = sha(path)
        for line, frame in read_rows(path):
            frames[(scene, frame["frame_idx"])] = (path, line, frame)
    motion = {(r["scene"], r["frame_idx"]): (line, r) for line, r in read_rows(MOTION)}
    result = []
    for line, row in false:
        path, frame_line, frame = frames[key(row)[:2]]
        node = frame["predictions"]["R1"][row["selected_node_id"]]
        match = next((m for m in frame["metrics"]["R1"]["matches"]
                      if m["prediction_index"] == row["selected_node_id"]), None)
        truth = frame["evaluation_only_official"][match["truth_index"]] if match else None
        if (truth["instance_token"] if truth else None) != row["evaluation_identity"]:
            raise AssertionError("冻结身份匹配不一致")
        motion_line, annotated = motion[key(row)[:2]]
        official = next((o for o in annotated["objects"]
                         if truth and o["annotation_token"] == truth["annotation_token"]), None)
        result.append({
            "scene": row["scene"], "frame_idx": row["frame_idx"], "query_index": row["query_index"],
            "query_text": row["query_text"], "old_false": old[key(row)]["selected"],
            "newly_false": not old[key(row)]["selected"], "R1_false": True,
            "evaluation_identity": row["evaluation_identity"],
            "predicted_node_id": node["node_id"], "predicted_category": node["category"],
            "predicted_motion_state": node.get("motion_state", "unknown"),
            "predicted_velocity_mps": node.get("velocity_mps"),
            "last_seen_frame": node.get("last_seen_frame"),
            "prediction_age_s": node.get("prediction_age_s"),
            "official_motion": official.get("official_motion") if official else None,
            "official_attributes": official.get("official_attributes") if official else None,
            "annotation_token": truth["annotation_token"] if truth else None,
            "evidence_status": "official_moving_vs_predicted_static" if official and
                official["official_motion"] == "dynamic" and node.get("motion_state") == "static"
                else "unmatched_identity_requires_case_audit" if truth is None else "other_matched_case",
            "selection_source": str(RAW), "selection_source_line": line,
            "frame_source": str(path), "frame_source_line": frame_line,
            "motion_source": str(MOTION), "motion_source_line": motion_line,
            "evaluation_only": True, "inference_changed": False,
        })
    additions = [r for r in result if r["newly_false"]]
    removed = sum(r["selected"] and not any(key(n) == k for _, n in false)
                  for k, r in old.items() if not r["positive"])
    summary = {"old_false": 8, "R1_false": len(result), "retained_old_false": len(result) - len(additions),
               "new_false": len(additions), "removed_old_false": removed,
               "new_official_moving_predicted_static": sum(r["evidence_status"] == "official_moving_vs_predicted_static"
                                                           for r in additions),
               "new_unmatched_identity": sum(r["evaluation_identity"] is None for r in additions),
               "new_model_or_selector_calls": 0,
               "limitation": "官方运动属性与预测静止不一致是诊断证据，尚未证明具体速度估计代码错误；unknown不擅自归因"}
    assert (summary["R1_false"], summary["retained_old_false"], summary["new_false"],
            summary["removed_old_false"], summary["new_official_moving_predicted_static"],
            summary["new_unmatched_identity"]) == (19, 8, 11, 0, 5, 6)
    if any(sha(Path(path)) != value for path, value in fingerprints.items()):
        raise AssertionError("诊断输入改变")
    with args.output.open("x") as handle:
        json.dump({"summary": summary, "input_sha256": fingerprints, "records": result}, handle,
                  ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
