"""独立复核固定GT视域、3m最大基数匹配与A/B/C主表，仅消费已存预测。"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.project_paths import PROJECT_ROOT
from multiview_20261123_prepare_inputs import digest, dump


def inside(center, view):
    xyz = np.linalg.solve(np.asarray(view["camera_pose"]), np.r_[center, 1.])[:3]
    uv = np.asarray(view["intrinsic"]) @ xyz
    return bool(1 < xyz[2] < 65 and 0 <= uv[0]/uv[2] < view["width"] and 0 <= uv[1]/uv[2] < view["height"])


def match(predictions, truth):
    m, n = len(predictions), len(truth)
    if not m or not n:
        return [], 0
    distance = np.full((m, n), np.inf)
    for i, p in enumerate(predictions):
        for j, t in enumerate(truth):
            if p["category"] == t["category"]:
                d = np.linalg.norm(np.asarray(p["center"][:2]) - t["center"][:2])
                if d <= 3:
                    distance[i, j] = d
    # Independent cost formulation: maximize legal cardinality then minimize sum distance.
    cost = np.zeros((m, n+m))
    cost[:, :n] = np.where(np.isfinite(distance), distance - 10000, 1e9)
    ii, jj = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if j < n and np.isfinite(distance[i, j])]
    used = {i for i, _ in pairs}
    duplicates = sum(i not in used and np.isfinite(distance[i]).any() for i in range(m))
    return pairs, int(duplicates)


def main():
    root = PROJECT_ROOT
    inputs = root / "dataset/processed/multiview_20261123_pilot_v1"
    run = root / "outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1"
    report = root / "outputs/reports/multiview_semantic_tasks_20261123"
    manifest = json.loads((inputs / "manifest.json").read_text())
    truth_path = inputs / "multiview_20261123_evaluation_ground_truth.jsonl"
    truth_rows = [json.loads(r) for r in truth_path.read_text().splitlines()]
    ground = {(r["scene"], r["frame_idx"]): r for r in truth_rows}
    assert len(ground) == len(truth_rows) == 40
    counts, side, gt_counts = defaultdict(Counter), defaultdict(Counter), Counter()
    views_audited = 0
    source_files = {str(inputs / "manifest.json"): digest(inputs / "manifest.json"), str(truth_path): digest(truth_path)}
    for scene in manifest["scenes"]:
        path = run / scene["scene"] / "multiview_20261123_predictions.jsonl.gz"
        source_files[str(path)] = digest(path)
        with gzip.open(path, "rt") as f:
            predictions = [json.loads(line) for line in f]
        assert len(predictions) == len(scene["groups"])
        for group, row in zip(scene["groups"], predictions):
            g = ground[(scene["scene"], group["frame_idx"])]
            assert g["sample_token"] == group["sample_token"] == row["sample_token"]
            union = g["objects"]
            assert len(union) == len({o["instance_token"] for o in union})
            for obj in union:
                actual_cameras = [c for c, v in group["views"].items() if inside(obj["center"], v)]
                assert actual_cameras == obj["visible_cameras"] and actual_cameras
                assert int(obj["visibility_token"]) >= 2
                assert obj["category"] in ("car", "truck", "bus", "motorcycle", "bicycle", "pedestrian")
            front = [o for o in union if "CAM_FRONT" in o["visible_cameras"]]
            gt_counts.update(union=len(union), front=len(front), side=len(union)-len(front))
            views_audited += len(group["views"])
            for method, all_predictions in row["predictions"].items():
                for scope, truth in (("front", front), ("union", union)):
                    cameras = {"CAM_FRONT": group["views"]["CAM_FRONT"]} if scope == "front" else group["views"]
                    pred = [p for p in all_predictions if any(inside(p["center"], v) for v in cameras.values())]
                    pairs, duplicates = match(pred, truth)
                    counts[(method, scope)].update(true_positive=len(pairs), truth=len(truth), predictions=len(pred), duplicate_nodes=duplicates)
                    if scope == "union":
                        sides = {i for i, o in enumerate(union) if "CAM_FRONT" not in o["visible_cameras"]}
                        matched = {j for _, j in pairs}
                        side[method].update(common_front_GT=len(front), common_front_TP=len(matched-sides), side_only_GT=len(sides), side_only_TP=len(matched & sides))
    result_path = report / "graph/multiview_20261123_result.json"
    expected = json.loads(result_path.read_text())["results"]
    for row in expected["node_main"]:
        c = counts[(row["method"], row["scope"])]
        assert all(c[k] == row[k] for k in c), (row["method"], row["scope"], c)
        assert row["false_negative"] == c["truth"] - c["true_positive"]
        assert row["unmatched_predictions"] == c["predictions"] - c["true_positive"]
        assert abs(row["recall"] - c["true_positive"]/c["truth"]) < 1e-12
        assert abs(row["precision"] - c["true_positive"]/c["predictions"]) < 1e-12
    for row in expected["side_coverage"]:
        assert dict(side[row["method"]]) == {k: row[k] for k in side[row["method"]]}
    assert dict(gt_counts) == dict(union=105, front=77, side=28)
    output = dict(passed=True, prediction_groups=40, calibrated_views=views_audited,
        GT_counts=dict(gt_counts), GT_unit="object-frame opportunities, instance-deduplicated within each camera group; not 105 unique objects",
        independent_fov_and_matching=True, matching="same category, XY<=3m, maximum legal cardinality then minimum total distance; all predictions retained by declared scope",
        new_model_calls=0, new_mapping_calls=0, official_annotation_full_table_rebuilt=False,
        limitation="Checks frozen per-frame GT FOV/membership/counts and independently rescored predictions; original official annotation completeness is separately attested by the GT producer.",
        node_counts=[dict(method=m, scope=s, **c) for (m, s), c in counts.items()],
        side_counts=[dict(method=m, **c) for m, c in side.items()],
        source_hashes=source_files, graph_result_sha256=digest(result_path), script_sha256=digest(__file__))
    target = report / "sensors/independent_graph_table_audit.json"
    dump(target, output)
    shutil.copy2(__file__, report / "sensors" / (Path(__file__).stem + "_source_snapshot.py"))
    print(json.dumps({k: v for k, v in output.items() if k not in ("source_hashes", "node_counts", "side_counts")}), flush=True)


if __name__ == "__main__":
    main()
