"""独立逐帧最大基数/距离匹配、固定TrackEval身份与序列配对区间核验。"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
import types
from pathlib import Path
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
RUN = ROOT / "statistics" / args.run_id
DEST = RUN / "independent_audit_002"
DEST.mkdir(exist_ok=False)
REF = ROOT / "source_bundle/identity_reference_001"
start = time.monotonic()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as h:
        json.dump(x, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write("\n")


def identity(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def pid(p, m, i):
    value = identity(p["node_id"])
    return f"frame={i};id={value}" if m == "single_frame" else value


def gate(g, p):
    return (
        g["category"] == p["category"]
        and math.hypot(g["center"][0] - p["center"][0], g["center"][1] - p["center"][1])
        <= 3
    )


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None, (a, b)
    else:
        assert math.isclose(a, b, abs_tol=1e-11, rel_tol=1e-11), (a, b)


def norm(c):
    pairs = {
        "IDR": ("IDTP", "GT_observations"),
        "IDP": ("IDTP", "prediction_observations"),
        "adjacent_identity_change_rate": (
            "adjacent_identity_changes",
            "adjacent_matched_transitions",
        ),
        "official_return_both_matched_rate": (
            "official_return_both_matched",
            "official_return_opportunities",
        ),
        "official_return_same_id_rate": (
            "official_return_same_id",
            "official_return_opportunities",
        ),
        "official_return_changed_id_rate": (
            "official_return_changed_id",
            "official_return_opportunities",
        ),
        "official_return_unknown_id_rate": (
            "official_return_unknown_id",
            "official_return_opportunities",
        ),
    }
    r = {k: c[a] / c[b] if c[b] else None for k, (a, b) in pairs.items()}
    n = c["GT_observations"] + c["prediction_observations"]
    r["IDF1"] = 2 * c["IDTP"] / n if n else None
    return r


def object_rates(c):
    pairs = {
        "recall": ("true_positive", "truth"),
        "matched_prediction_fraction": ("true_positive", "predictions"),
        "center_error_m": ("center_error_sum_m", "true_positive"),
        "duplicate_fraction": ("duplicate_nodes", "predictions"),
    }
    return {k: c[a] / c[b] if c[b] else None for k, (a, b) in pairs.items()}


def ci(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return (np.quantile(v, [0.025, 0.975]).tolist() if len(v) else None), len(v)


try:
    report = json.loads((RUN / "report.json").read_text())
    scene_reports = json.loads((RUN / "scene_reports.json").read_text())
    rel = json.loads((RUN / "release.json").read_text())
    mapping = ROOT / "mapping_evaluation" / rel["mapping_run"]
    assert (
        sha(mapping / "release.json") == rel["mapping_release_sha256"]
        and sha(mapping / "summary.json") == rel["mapping_summary_sha256"]
    )
    manifest = json.loads((REF / "manifest.json").read_text())
    compat = DEST / "reference_compatibility"
    changes = []
    for row in manifest["files"]:
        path = REF / "original" / row["path"]
        assert sha(path) == row["sha256"]
        raw = path.read_text()
        converted = (
            raw.replace(".astype(np.int)", ".astype(int)")
            if row["path"] == "trackeval/metrics/identity.py"
            else raw
        )
        p = compat / row["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(converted)
        if converted != raw:
            changes.append(
                dict(
                    path=row["path"],
                    replacements=raw.count(".astype(np.int)"),
                    original_sha256=sha(path),
                    compatibility_sha256=sha(p),
                )
            )
    for name, relpath in [
        ("trackeval", "trackeval"),
        ("trackeval.metrics", "trackeval/metrics"),
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(compat / relpath)]
        sys.modules[name] = mod
    reference = importlib.import_module("trackeval.metrics.identity").Identity(
        dict(THRESHOLD=0.5, PRINT_CONFIG=False)
    )
    save(
        DEST / "release.json",
        dict(
            official_commit=manifest["commit"],
            reference_sha256=sha(REF / "manifest.json"),
            compatibility_changes=changes,
            audit_source_sha256=sha(__file__),
            wall_budget_s=600,
            minimum_free_gb=6,
            independent_frame_matching="NetworkX max_weight_matching(maxcardinality=True), edge weight=-hypot",
            goal_complete=False,
        ),
    )
    methods = sorted(report["object"]["methods"])
    supports = ["all_gated", "mutually_unique_gated"]
    checked = Counter()
    independent = {}
    objseq = {}
    events = {}
    sourcehash = {}
    for p, expected in report["source_evaluation_frames_sha256"].items():
        assert (
            time.monotonic() - start < 600
            and shutil.disk_usage(ROOT).free > 6 * 1024**3
        )
        assert sha(p) == expected
        sourcehash[p] = expected
        frames = [json.loads(x) for x in open(p)]
        seq = frames[0]["sequence"]
        assert [f["frame_idx"] for f in frames] == list(range(len(frames)))
        checked["frames"] += len(frames)
        checked["sequences"] += 1
        for method in methods:
            totals = Counter()
            visible = defaultdict(list)
            matches = []
            oldmatches = []
            node_instances = defaultdict(set)
            for i, f in enumerate(frames):
                gt = f["evaluation_only_official"]
                pred = f["predictions"][method]
                graph = nx.Graph()
                for gi, g in enumerate(gt):
                    visible[identity(g["instance_token"])].append(i)
                    for pi, prediction in enumerate(pred):
                        if gate(g, prediction):
                            graph.add_edge(
                                ("g", gi),
                                ("p", pi),
                                weight=-math.hypot(
                                    g["center"][0] - prediction["center"][0],
                                    g["center"][1] - prediction["center"][1],
                                ),
                            )
                assignment = nx.max_weight_matching(graph, maxcardinality=True)
                expected_n = len(assignment)
                expected_d = -sum(graph[a][b]["weight"] for a, b in assignment)
                actual = f["metrics"][method]
                assert actual["true_positive"] == expected_n
                close(sum(x["distance_m"] for x in actual["matches"]), expected_d)
                seen_gt = set()
                seen_pred = set()
                matched = {}
                old = {}
                for x in actual["matches"]:
                    gi, pi = x["truth_index"], x["prediction_index"]
                    assert (
                        gi not in seen_gt
                        and pi not in seen_pred
                        and gate(gt[gi], pred[pi])
                    )
                    seen_gt.add(gi)
                    seen_pred.add(pi)
                    token = identity(gt[gi]["instance_token"])
                    matched[token] = pid(pred[pi], method, i)
                    old[token] = identity(pred[pi]["node_id"])
                    node_instances[old[token]].add(token)
                duplicates = sum(
                    j not in seen_pred and any(gate(g, p) for g in gt)
                    for j, p in enumerate(pred)
                )
                counts = dict(
                    predictions=len(pred),
                    truth=len(gt),
                    true_positive=expected_n,
                    unmatched_predictions=len(pred) - expected_n,
                    false_negative=len(gt) - expected_n,
                    duplicate_nodes=duplicates,
                )
                for k, v in counts.items():
                    assert actual[k] == v
                counts["center_error_sum_m"] = expected_d
                totals.update(counts)
                matches.append(matched)
                oldmatches.append(old)
                checked["frame_method_metrics"] += 1
            objseq[seq, method] = totals
            for k, v in scene_reports[seq][method]["object"].items():
                close(v, totals[k])
            returns = Counter()
            actual_return = scene_reports[seq][method]["return_diagnostics"]
            return_events = {}
            tails = {}
            for token, indices in visible.items():
                returns["GT_instances"] += 1
                matched_indices = [i for i in indices if token in matches[i]]
                for a, b in zip(indices, indices[1:]):
                    if b > a + 1:
                        left, right = matches[a].get(token), matches[b].get(token)
                        status = (
                            "unknown"
                            if left is None or right is None
                            else ("same" if left == right else "changed")
                        )
                        returns["official_return_opportunities"] += 1
                        returns["official_return_both_matched"] += status != "unknown"
                        returns[f"official_return_{status}_id"] += 1
                        return_events["official_list_return", token, a, b] = status
                for a, b in zip(matched_indices, matched_indices[1:]):
                    if b == a + 1:
                        returns["adjacent_matched_transitions"] += 1
                        returns["adjacent_identity_changes"] += (
                            oldmatches[a][token] != oldmatches[b][token]
                        )
                    else:
                        returns["method_gap_returns"] += 1
                        status = (
                            "same"
                            if matches[a][token] == matches[b][token]
                            else "changed"
                        )
                        returns[
                            "method_gap_same_id"
                            if status == "same"
                            else "method_gap_changed_id"
                        ] += 1
                        return_events["method_matched_gap_return", token, a, b] = status
                if not matched_indices:
                    returns["never_matched_instances"] += 1
                    returns["never_matched_GT_observations"] += len(indices)
                elif matched_indices[-1] < len(frames) - 1:
                    returns["censored_tails"] += 1
                    returns["tails_GT_absent_at_scene_end"] += (
                        indices[-1] < len(frames) - 1
                    )
                    returns["unreturned_visible_tail_observations"] += sum(
                        i > matched_indices[-1] for i in indices
                    )
                    tails[token] = (
                        matched_indices[-1],
                        indices[-1],
                        sum(i > matched_indices[-1] for i in indices),
                    )
            for k, v in actual_return["counts"].items():
                assert v == returns[k], (seq, method, k, v, returns[k])
            actual_events = {
                (x["kind"], x["GT_instance"], x["left_frame"], x["right_frame"]): x[
                    "identity_status"
                ]
                for x in actual_return["events"]
            }
            assert return_events == actual_events
            actual_tails = {
                x["GT_instance"]: (
                    x["last_matched_frame"],
                    x["last_GT_visible_frame"],
                    x["unmatched_visible_observations"],
                )
                for x in actual_return["censored_tails"]
            }
            assert actual_tails == tails
            checked["return_events"] += len(return_events)
            checked["censored_tails"] += len(tails)
            legacy = scene_reports[seq][method]["legacy_identity"]
            assert (
                legacy["identity_switches"] == returns["adjacent_identity_changes"]
                and legacy["adjacent_matched_transitions"]
                == returns["adjacent_matched_transitions"]
            )
            assert legacy["nodes_associated_with_multiple_instances"] == sum(
                len(x) > 1 for x in node_instances.values()
            )
            gids = sorted(visible)
            pids = sorted(
                {
                    pid(o, method, i)
                    for i, f in enumerate(frames)
                    for o in f["predictions"][method]
                }
            )
            gi = {x: i for i, x in enumerate(gids)}
            pi = {x: i for i, x in enumerate(pids)}
            for support in supports:
                count = Counter()
                glist = []
                plist = []
                similarities = []
                potential = Counter()
                for i, f in enumerate(frames):
                    gt = f["evaluation_only_official"]
                    pred = f["predictions"][method]
                    valid = np.array(
                        [[gate(g, p) for p in pred] for g in gt], dtype=bool
                    ).reshape(len(gt), len(pred))
                    dg = valid.sum(1)
                    dp = valid.sum(0)
                    matrix = (
                        valid
                        if support == "all_gated"
                        else valid & (dg[:, None] == 1) & (dp[None, :] == 1)
                    )
                    count.update(
                        dict(
                            GT_observations=len(gt),
                            prediction_observations=len(pred),
                            ambiguous_GT_observations=int((dg > 1).sum()),
                            ambiguous_prediction_observations=int((dp > 1).sum()),
                            gated_edges=int(valid.sum()),
                            supported_edges=int(matrix.sum()),
                        )
                    )
                    for g, pred_index in zip(*np.nonzero(matrix)):
                        potential[
                            identity(gt[g]["instance_token"]),
                            pid(pred[pred_index], method, i),
                        ] += 1
                    glist.append(
                        np.array(
                            [gi[identity(g["instance_token"])] for g in gt], dtype=int
                        )
                    )
                    plist.append(
                        np.array([pi[pid(p, method, i)] for p in pred], dtype=int)
                    )
                    similarities.append(matrix.astype(float))
                data = dict(
                    num_gt_dets=count["GT_observations"],
                    num_tracker_dets=count["prediction_observations"],
                    num_gt_ids=len(gids),
                    num_tracker_ids=len(pids),
                    gt_ids=glist,
                    tracker_ids=plist,
                    similarity_scores=similarities,
                )
                official = reference.eval_sequence(data)
                for k in ["IDTP", "IDFN", "IDFP"]:
                    count[k] = int(official[k])
                actual = json.loads(
                    (RUN / f"{seq}_{method}_{support}.json").read_text()
                )
                for k, v in actual["counts"].items():
                    assert count[k] == v, (seq, method, support, k)
                assert (
                    len({x["GT_instance"] for x in actual["assignments"]})
                    == len(actual["assignments"])
                    == len({x["prediction_identity"] for x in actual["assignments"]})
                )
                assert (
                    sum(x["overlap_observations"] for x in actual["assignments"])
                    == count["IDTP"]
                )
                for x in actual["assignments"]:
                    assert (
                        potential[x["GT_instance"], x["prediction_identity"]]
                        == x["overlap_observations"]
                    )
                count.update(returns)
                independent[seq, method, support] = count
                checked["identity_cases"] += 1
    sequences = sorted({k[0] for k in objseq})
    draws = np.random.default_rng(17).integers(
        len(sequences), size=(4000, len(sequences))
    )
    boot = {}
    estimates = {}
    for method in methods:
        total = sum((objseq[s, method] for s in sequences), Counter())
        actual = report["object"]["methods"][method]
        for k, v in actual["counts"].items():
            close(total[k], v)
        for k, v in object_rates(total).items():
            close(actual["metrics"][k], v)
        ob = defaultdict(list)
        for d in draws:
            vals = object_rates(
                sum((objseq[sequences[i], method] for i in d), Counter())
            )
            for k, v in vals.items():
                ob[k].append(np.nan if v is None else v)
        boot["object", method] = {k: np.array(v) for k, v in ob.items()}
        estimates["object", method] = object_rates(total)
        for support in supports:
            total = sum((independent[s, method, support] for s in sequences), Counter())
            actual = report["identity"]["methods"][method][support]
            for k, v in actual["counts"].items():
                assert total[k] == v
            estimates[method, support] = norm(total)
            ib = defaultdict(list)
            for d in draws:
                vals = norm(
                    sum(
                        (independent[sequences[i], method, support] for i in d),
                        Counter(),
                    )
                )
                for k, v in vals.items():
                    ib[k].append(np.nan if v is None else v)
            boot[method, support] = {k: np.array(v) for k, v in ib.items()}
            for k, v in norm(total).items():
                recorded = actual["metrics"][k]
                close(recorded["value"], v)
                interval, n = ci(boot[method, support][k])
                assert recorded["ci95"] == interval and recorded["valid_resamples"] == n
                checked["identity_intervals"] += 1
    for family in ["object", "identity"]:
        for row in report[family]["paired_differences"]:
            method = row["method"]
            metric = row["metric"]
            key = ("object", method) if family == "object" else (method, row["support"])
            base = (
                ("object", "original_fusion")
                if family == "object"
                else ("original_fusion", row["support"])
            )
            a, b = estimates[key][metric], estimates[base][metric]
            close(row["difference"], a - b if a is not None and b is not None else None)
            interval, n = ci(boot[key][metric] - boot[base][metric])
            assert row["valid_resamples"] == n
            if interval is None:
                assert row["ci95"] is None
            else:
                assert np.allclose(row["ci95"], interval, rtol=0, atol=1e-12), (
                    family,
                    row,
                    interval,
                )
                checked["floating_interval_max_abs_difference_femtounits"] = max(
                    checked["floating_interval_max_abs_difference_femtounits"],
                    int(
                        round(
                            float(np.max(np.abs(np.asarray(row["ci95"]) - interval)))
                            * 1e15
                        )
                    ),
                )
            checked[family + "_paired_intervals"] += 1
    assert (
        checked["frames"] == report["frame_count"]
        and checked["sequences"] == report["sequence_count"]
    )
    save(
        DEST / "report.json",
        dict(
            passed=True,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            checks=dict(checked),
            independent_frame_assignment=True,
            TrackEval_counts_equal=True,
            all_denominators_retained=True,
            paired_sequence_bootstrap_recomputed=True,
            source_sha256=sourcehash,
            wall_s=time.monotonic() - start,
            official_KITTI_protocol=False,
            navigation_arrivals=0,
            goal_complete=False,
        ),
    )
    print(json.dumps(dict(passed=True, checks=dict(checked))))
except BaseException as exc:
    save(DEST / "interrupted.json", dict(error=repr(exc), goal_complete=False))
    raise
