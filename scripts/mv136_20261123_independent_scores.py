"""独立重算saved predictions的固定3m节点/身份/失得，不调用建图或模型。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def in_view(center, view):
    point = np.linalg.solve(np.asarray(view["camera_pose"]), np.r_[center, 1.])[:3]
    uvz = np.asarray(view["intrinsic"]) @ point
    return bool(1 < point[2] < 65 and 0 <= uvz[0]/uvz[2] < view["width"] and 0 <= uvz[1]/uvz[2] < view["height"])


def identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError("无效ID；不能删去异常预测提高成绩")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def score_frame(pred, truth):
    ng, npred = len(truth), len(pred)
    distance = np.full((ng, npred), np.inf)
    for g, obj in enumerate(truth):
        for p, item in enumerate(pred):
            if obj["category"] == item["category"]:
                d = np.linalg.norm(np.asarray(obj["center"][:2]) - item["center"][:2])
                if d <= 3:
                    distance[g, p] = d
    gate = np.isfinite(distance)
    if ng and npred:
        cost = np.zeros((ng, npred + ng))
        cost[:, :npred] = np.where(gate, distance - 10000., 1e9)
        gi, pi = linear_sum_assignment(cost)
        pairs = [(int(g), int(p)) for g, p in zip(gi, pi) if p < npred and gate[g, p]]
    else:
        pairs = []
    used = {p for _, p in pairs}
    duplicate = sum(p not in used and gate[:, p].any() for p in range(npred))
    counts = Counter(TP=len(pairs), GT=ng, predictions=npred, FP=npred-len(pairs), FN=ng-len(pairs), duplicates=int(duplicate))
    counts["error_sum_m"] = sum(float(distance[g, p]) for g, p in pairs)
    return counts, pairs, gate


def score_identity(records):
    gt_ids = sorted({identity(o["instance_token"]) for r in records for o in r["truth"]})
    pred_ids = sorted({identity(o["node_id"]) for r in records for o in r["pred"]})
    gi, pi = {g:i for i,g in enumerate(gt_ids)}, {p:i for i,p in enumerate(pred_ids)}
    potential = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    unique_potential = np.zeros_like(potential)
    ambiguity_gt = ambiguity_pred = 0
    matched_by_gt = defaultdict(list)
    matched_by_pred = defaultdict(set)
    for row in records:
        gate, truth, pred = row["gate"], row["truth"], row["pred"]
        gd, pd = gate.sum(1), gate.sum(0)
        ambiguity_gt += int((gd > 1).sum())
        ambiguity_pred += int((pd > 1).sum())
        for g, p in zip(*np.nonzero(gate)):
            potential[gi[identity(truth[g]["instance_token"])], pi[identity(pred[p]["node_id"])]] += 1
            if gd[g] == 1 and pd[p] == 1:
                unique_potential[gi[identity(truth[g]["instance_token"])], pi[identity(pred[p]["node_id"])]] += 1
        for g, p in row["pairs"]:
            gid, pid = identity(truth[g]["instance_token"]), identity(pred[p]["node_id"])
            matched_by_gt[gid].append((row["frame_idx"], pid))
            matched_by_pred[pid].add(gid)
    a,b = linear_sum_assignment(-potential)
    au,bu = linear_sum_assignment(-unique_potential)
    total_gt = sum(len(r["truth"]) for r in records)
    total_pred = sum(len(r["pred"]) for r in records)
    tp = int(potential[a,b].sum())
    unique_tp = int(unique_potential[au,bu].sum())
    adjacent = changes = 0
    for matches in matched_by_gt.values():
        for (f1,p1),(f2,p2) in zip(matches,matches[1:]):
            if f2 == f1 + 1:
                adjacent += 1
                changes += p1 != p2
    assert 0 <= tp <= min(total_gt,total_pred)
    return Counter(IDTP=tp, IDFN=total_gt-tp, IDFP=total_pred-tp, unique_gated_IDTP=unique_tp,
        identity_GT=total_gt, identity_predictions=total_pred, ambiguous_GT=ambiguity_gt,
        ambiguous_predictions=ambiguity_pred, adjacent_matched_transitions=adjacent,
        adjacent_ID_changes=changes, GT_with_multiple_matched_IDs=sum(len({p for _,p in v})>1 for v in matched_by_gt.values()),
        IDs_matched_to_multiple_GT=sum(len(v)>1 for v in matched_by_pred.values()), GT_identities=len(gt_ids), predicted_identities=len(pred_ids))


def percentages(c):
    def ratio(a,b):
        return a/b if b else None
    return dict(precision=ratio(c["TP"],c["predictions"]), recall=ratio(c["TP"],c["GT"]),
        F1=ratio(2*c["TP"],c["GT"]+c["predictions"]), IDF1=ratio(2*c["IDTP"],c["GT"]+c["predictions"]),
        unique_gated_IDF1=ratio(2*c["unique_gated_IDTP"],c["GT"]+c["predictions"]),
        duplicates_per_prediction=ratio(c["duplicates"],c["predictions"]), duplicates_per_GT=ratio(c["duplicates"],c["GT"]),
        mean_center_error_m=ratio(c["error_sum_m"],c["TP"]))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=path; multiple labels are independently pooled by scene")
    parser.add_argument("--reference", action="append", default=[], help="label:method")
    parser.add_argument("--paired-method", action="append", default=[], help="explicit A/C/candidate keys for common-target position diagnostics")
    parser.add_argument("--prediction-pattern", help="relative path with {scene}; otherwise find one predictions file under scene")
    parser.add_argument("--out-id", required=True)
    parser.add_argument("--expect-pilot", action="store_true")
    args = parser.parse_args()
    output = experiment_directory("nuscenes", "validation", "mv136_independent_"+args.out_id, create=True)
    report = PROJECT_ROOT / "outputs/reports/multiview_viewcount_20261123/independent"
    report.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    shutil.copy2(__file__, output / "source_snapshot" / Path(__file__).name)
    manifest = json.loads(args.manifest.read_text())
    truth_rows = [json.loads(x) for x in args.ground_truth.read_text().splitlines()]
    ground = {(r["scene"],r["frame_idx"]):r for r in truth_rows}
    expected = {(s["scene"],g["frame_idx"]) for s in manifest["scenes"] for g in s["groups"]}
    assert set(ground) == expected and len(ground) == len(truth_rows)
    hashes = {str(args.manifest):digest(args.manifest), str(args.ground_truth):digest(args.ground_truth)}
    totals, records, cases, frames = defaultdict(Counter), defaultdict(list), {}, []
    side, leaked_tokens, per_scene = defaultdict(Counter), Counter(), []
    for declared in args.run:
        label, raw = declared.split("=",1)
        root = Path(raw)
        methods_seen = None
        for scene in manifest["scenes"]:
            name = scene["scene"]
            found = [root / args.prediction_pattern.format(scene=name)] if args.prediction_pattern else list((root/name).glob("*predictions*.jsonl.gz"))
            if len(found) != 1:
                raise ValueError(f"{name}: expected one predictions file, got {found}")
            path = found[0]
            hashes[str(path)] = digest(path)
            with gzip.open(path,"rt") as f:
                rows = [json.loads(line) for line in f]
            assert len(rows) == len(scene["groups"])
            local = defaultdict(Counter)
            for group,row in zip(scene["groups"],rows):
                assert row["frame_idx"] == group["frame_idx"]
                if "sample_token" in row:
                    assert row["sample_token"] == group["sample_token"]
                assert row.get("GT_in_inference",False) is False
                official = ground[(name,group["frame_idx"])]
                assert official["sample_token"] == group["sample_token"]
                union = official["objects"]
                assert len(union) == len({o["instance_token"] for o in union})
                front = [o for o in union if "CAM_FRONT" in o["visible_cameras"]]
                front3 = [o for o in union if set(o["visible_cameras"]) & {"CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT"}]
                for obj in union:
                    assert obj["visible_cameras"] == [c for c,v in group["views"].items() if in_view(obj["center"],v)]
                if methods_seen is None:
                    methods_seen = set(row["predictions"])
                assert methods_seen == set(row["predictions"])
                for method,all_pred in row["predictions"].items():
                    key = label+":"+method
                    assert len(all_pred) == len({identity(p["node_id"]) for p in all_pred}), "同帧重复ID不能用于身份信用"
                    for p in all_pred:
                        assert np.asarray(p["center"]).shape == (3,) and np.isfinite(p["center"]).all()
                        assert p["category"] in ("car","truck","bus","bicycle","motorcycle","pedestrian")
                        leaked_tokens[key] += str(p["node_id"]) in {o["instance_token"] for o in union}
                    for scope,truth in (("front",front),("front3",front3),("union6",union)):
                        views = {c:v for c,v in group["views"].items() if c=="CAM_FRONT" or scope=="union6" or (scope=="front3" and c in ("CAM_FRONT_LEFT","CAM_FRONT_RIGHT"))}
                        pred = [p for p in all_pred if any(in_view(p["center"],v) for v in views.values())]
                        count,pairs,gate = score_frame(pred,truth)
                        local[key,scope].update(count)
                        rec = dict(scene=name,frame_idx=group["frame_idx"],truth=truth,pred=pred,gate=gate,pairs=pairs)
                        records[key,scope,name].append(rec)
                        matches = {truth[g]["instance_token"]:pred[p] for g,p in pairs}
                        cases[key,scope,name,group["frame_idx"]] = dict(matches=matches,pred=pred)
                        frames.append(dict(source=label,method=method,scene=name,frame_idx=group["frame_idx"],scope=scope,**count))
                        if scope == "union6":
                            front_ids = {o["instance_token"] for o in front}
                            side[key].update(front_GT=len(front),side_GT=len(union)-len(front),
                                front_TP=sum(t in front_ids for t in matches),side_TP=sum(t not in front_ids for t in matches),
                                front3_GT=len(front3),rear_only_GT=len(union)-len(front3),
                                front3_TP=sum(t in {o["instance_token"] for o in front3} for t in matches),
                                rear_only_TP=sum(t not in {o["instance_token"] for o in front3} for t in matches))
            for (key,scope),count in local.items():
                count.update(score_identity(records[key,scope,name]))
                assert count["IDTP"] <= count["TP"]
                totals[key,scope].update(count)
                per_scene.append(dict(method=key,scope=scope,scene=name,log_token=scene["log_token"],**count,**percentages(count)))
    summary = [dict(method=key,scope=scope,**count,**percentages(count)) for (key,scope),count in totals.items()]
    if args.expect_pilot:
        assert all(c["GT"] == (77 if scope == "front" else 105) for (key,scope),c in totals.items())
    changes, details = [],[]
    keys = sorted({key for key,scope in totals})
    for reference in args.reference:
        assert reference in keys
        for candidate in keys:
            if candidate == reference:
                continue
            for scope in ("front","front3","union6"):
                gained = lost = preserved = added_pred = removed_pred = 0
                solely_removed = same_geometry = True
                for scene in manifest["scenes"]:
                    for group in scene["groups"]:
                        scene_name,idx=scene["scene"],group["frame_idx"]
                        before = cases[reference,scope,scene_name,idx]
                        after = cases[candidate,scope,scene_name,idx]
                        a,b = set(before["matches"]),set(after["matches"])
                        gained += len(b-a);lost += len(a-b);preserved += len(a&b)
                        for kind,instances in (("gain",b-a),("loss",a-b)):
                            front_ids={o["instance_token"] for o in ground[scene_name,idx]["objects"] if "CAM_FRONT" in o["visible_cameras"]}
                            for token in sorted(instances):
                                details.append(dict(reference=reference,candidate=candidate,scope=scope,scene=scene_name,frame_idx=idx,change=kind,
                                    GT_instance=token,GT_sector="front" if token in front_ids else "side_only",
                                    previous_prediction=before["matches"].get(token),candidate_prediction=after["matches"].get(token)))
                        def signature(p):
                            return (p["category"],*(round(float(x),6) for x in p["center"]))
                        sa,sb=Counter(map(signature,before["pred"])),Counter(map(signature,after["pred"]))
                        added_pred += sum((sb-sa).values());removed_pred += sum((sa-sb).values())
                        solely_removed &= not bool(sb-sa);same_geometry &= sa == sb
                changes.append(dict(reference=reference,candidate=candidate,scope=scope,gained_GT_frames=gained,lost_GT_frames=lost,
                    preserved_GT_frames=preserved,net_TP=gained-lost,added_geometry_predictions=added_pred,removed_geometry_predictions=removed_pred,
                    only_removed_predictions=bool(solely_removed),geometry_identical_ignoring_ID=bool(same_geometry)))
    paired_methods = args.paired_method or keys
    assert len(paired_methods) == len(set(paired_methods)) and all(m in keys for m in paired_methods)
    paired_samples, paired_steps, paired_summary = [], [], []
    for scope in ("front", "front3", "union6"):
        for scene in manifest["scenes"]:
            previous = {}
            for group in scene["groups"]:
                name, idx = scene["scene"], group["frame_idx"]
                common = set.intersection(*(set(cases[m,scope,name,idx]["matches"]) for m in paired_methods))
                truth_by_id = {o["instance_token"]:o for o in ground[name,idx]["objects"]}
                current = {}
                for token in sorted(common):
                    point = np.asarray(truth_by_id[token]["center"][:2])
                    errors = {m:np.asarray(cases[m,scope,name,idx]["matches"][token]["center"][:2])-point for m in paired_methods}
                    for method,error in errors.items():
                        paired_samples.append(dict(scope=scope,scene=name,frame_idx=idx,GT_instance=token,method=method,
                            error_x_m=float(error[0]),error_y_m=float(error[1]),center_error_xy_m=float(np.linalg.norm(error)),
                            official_sample_timestamp_s=group["official_sample_timestamp_s"],group_reference_timestamp_s=group["reference_timestamp_s"]))
                    if token in previous:
                        prev = previous[token]
                        if prev["frame_idx"] == idx - 1:
                            dt = group["official_sample_timestamp_s"] - prev["sample_time_s"]
                            assert dt > 0
                            for method,error in errors.items():
                                step = error-prev["errors"][method]
                                paired_steps.append(dict(scope=scope,scene=name,previous_frame=idx-1,frame_idx=idx,GT_instance=token,method=method,
                                    delta_error_norm_m=float(np.linalg.norm(step)),sample_delta_time_s=dt,
                                    definition="||(prediction_t-GT_t)-(prediction_prev-GT_prev)||; removes real GT object displacement"))
                    current[token] = dict(frame_idx=idx,sample_time_s=group["official_sample_timestamp_s"],errors=errors)
                previous = current
        for method in paired_methods:
            position = [r["center_error_xy_m"] for r in paired_samples if r["scope"] == scope and r["method"] == method]
            steps = [r["delta_error_norm_m"] for r in paired_steps if r["scope"] == scope and r["method"] == method]
            paired_summary.append(dict(scope=scope,method=method,common_object_frames=len(position),common_adjacent_object_pairs=len(steps),
                common_center_error_mean_m=float(np.mean(position)) if position else None,
                common_center_error_median_m=float(np.median(position)) if position else None,
                common_center_error_RMSE_m=float(np.sqrt(np.mean(np.square(position)))) if position else None,
                common_error_change_mean_m=float(np.mean(steps)) if steps else None,
                common_error_change_median_m=float(np.median(steps)) if steps else None,
                common_error_change_p90_m=float(np.quantile(steps,.9)) if steps else None))
    position_note = "Only intersection of matched identical GT object-frames across declared A/C/candidate methods; adjacent metric subtracts true GT displacement. Official GT is at sample time, prediction at camera/group time; no dynamic GT motion compensation. Full denominators and lost coverage remain in main/changes tables. Small common N does not establish stability generalization."
    result = dict(passed=True,manifest=str(args.manifest),ground_truth=str(args.ground_truth),threshold_m=3.,distance="XY same-category",
        grouping="identity assigned one-to-one per scene, all frame GT and prediction denominators retained",new_model_calls=0,new_mapping_calls=0,
        source_sha256=digest(__file__),input_sha256=hashes,groups=len(expected),logs=len({s["log_token"] for s in manifest["scenes"]}),
        summaries=summary,side_counts=[dict(method=k,**v) for k,v in side.items()],changes=changes,
        paired_position_methods=paired_methods,paired_position_summary=paired_summary,paired_position_limitations=position_note,
        IDs_equal_official_instance_tokens=dict(leaked_tokens),
        limitations="IDF1 uses fixed all-gated global evaluator matching, not inference-time GT ID remapping. This audit cannot prove source causality; candidate source/config/GT isolation must also be inspected. No unseen-test claim.")
    (output / "scores.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (output / "gain_loss_cases.json").write_text(json.dumps(details,ensure_ascii=False,indent=2)+"\n")
    (output / "paired_position_samples.json").write_text(json.dumps(dict(samples=paired_samples,adjacent_pairs=paired_steps,limitations=position_note),ensure_ascii=False,indent=2)+"\n")
    for name,rows in (("main",summary),("scene",per_scene),("frame",frames),("changes",changes),("side",result["side_counts"])):
        write_csv(report / (args.out_id+"_"+name+".csv"),rows)
    write_csv(report / (args.out_id+"_paired_position.csv"),paired_summary)
    (report / (args.out_id+"_audit.json")).write_text(json.dumps(dict(passed=True,output=str(output),scores_sha256=digest(output/"scores.json"),summary=summary,changes=changes),ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(passed=True,output=str(output),rows=len(summary),groups=len(expected),new_model_calls=0,new_mapping_calls=0)),flush=True)


if __name__ == "__main__":
    main()
