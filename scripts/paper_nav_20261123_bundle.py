"""Aggregate frozen navigation evidence and prepare an unscored episode input bundle.

No selector, model, planner, or vehicle is executed by this command.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

PROJECT = Path(__file__).resolve().parents[1]
LEGACY = PROJECT / "outputs/project_v2/thesis_completion_v1"
ITERATIONS = PROJECT / "outputs/results/qwen/validation/iterations"
TRANSFER = ITERATIONS / "selection_r1_map_transfer_20261123_v1"
RELATION = ITERATIONS / "selection_relation_support_20261123_v1"
RETRIEVAL = LEGACY / "main_experiments/causal_retrieval_full_001"
GOALS = LEGACY / "ablations/causal_goals_full_001"
WORKPOINTS = ITERATIONS / "selection_rejection_workpoints_20261123_v1"
EPISODE = PROJECT / "dataset/annotations/natural_graph_v004/episode_scene0012_20261120_001"
R1_FRAME = PROJECT / ("outputs/results/nuscenes/validation/iterations/"
                     "recall_refinement_R1_full_20261120_v2/scene-0012/frames.jsonl.gz")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for index, line in enumerate(handle, 1):
            yield index, json.loads(line)


def dump(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path, values):
    with Path(path).open("x") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, values):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def key(row):
    return row["scene"], row["frame_idx"], row["query_index"]


def percent(numerator, denominator):
    return 100 * numerator / denominator if numerator is not None and denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    state = read(args.run_dir / "start_manifest.json")
    if shutil.disk_usage(PROJECT).free < state["min_free_bytes"]:
        raise OSError("磁盘低于20GiB保护线")
    args.input_dir.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    release = read(RETRIEVAL / "release.json")
    frame_paths = [Path(path) for path in release["source_frames_sha256"]]
    inputs = [TRANSFER / "queries.jsonl", TRANSFER / "report.json",
              RELATION / "queries.jsonl", RELATION / "report.json",
              RETRIEVAL / "release.json", RETRIEVAL / "queries.jsonl",
              GOALS / "queries.jsonl", GOALS / "release.json", GOALS / "summary.json",
              WORKPOINTS / "report.json", EPISODE / "manifest.json",
              EPISODE / "causal_graphs.jsonl", R1_FRAME, *frame_paths]
    fingerprints = {str(path): sha(path) for path in inputs}
    for path, expected in release["source_frames_sha256"].items():
        if fingerprints[path] != expected:
            raise ValueError("历史地图输入指纹不匹配")
    dump(args.run_dir / "frozen_configuration.json", {
        "purpose": "既有结果再汇总与无成绩连续协议准备，不运行选择器或规划器",
        "input_sha256": fingerprints, "source_sha256": {str(Path(__file__)): sha(__file__)},
        "queries": 1115, "positive": 733, "negative": 382,
        "evaluation_match_xy_m": 3.0, "new_model_calls": 0,
        "new_experiment_or_parameter_search": False, "random_sampling": False,
        "episode_labels": "pending_real_human_review", "budget": state,
    })

    transfer = list(rows(TRANSFER / "queries.jsonl"))
    old_rows = {key(r): r for _, r in transfer if r["method"] == "old_map_original_rule"}
    associations = {}
    old_frames = {}
    query_frames = {k[:2] for k in old_rows}
    for path in frame_paths:
        for line, frame in rows(path):
            frame_key = frame["scene"], frame["frame_idx"]
            if frame_key in query_frames:
                gt = frame["evaluation_only_official"]
                associations[frame_key] = {
                    match["prediction_index"]: gt[match["truth_index"]]["instance_token"]
                    for match in frame["metrics"]["one_to_one_depth_quality"]["matches"]}
            if frame["scene"] == "scene-0012":
                old_frames[frame["frame_idx"]] = (line, frame, path)
    if len(associations) != 382 or len(old_rows) != 1115:
        raise AssertionError("必须保留382画面和1115任务")
    groups = {}
    for method in ("old_map_original_rule", "R1_map_original_rule"):
        groups[method] = [(line, r, TRANSFER / "queries.jsonl")
                          for line, r in transfer if r["method"] == method]
    for line, record in rows(RELATION / "queries.jsonl"):
        if record["method"] != "full_joint_rule":
            groups.setdefault(record["method"], []).append((line, record, RELATION / "queries.jsonl"))

    names = {
        "old_map_original_rule": "旧累积图＋原完整规则",
        "R1_map_original_rule": "R1累积图＋同一原完整规则",
        "relation_support_consistent": "旧图＋关系范围一致性修正",
        "without_relation_edges": "旧图＋去持久关系边（保留几何）",
        "without_depth_evidence_gate": "旧图＋去选择证据门（历史消融）",
        "category_nearest_current_detection": "当前帧类别最近检测",
        "clip_current_crop": "当前帧CLIP裁剪检索",
        "vlmaps_style_sparse_clip_grid": "稀疏CLIP网格检索",
    }
    normalized = []
    summaries = []
    order = list(names)
    transfer_report = read(TRANSFER / "report.json")
    relation_report = read(RELATION / "report.json")
    for method in order:
        evidence = groups[method]
        if {key(r) for _, r, _ in evidence} != set(old_rows) or len(evidence) != 1115:
            raise AssertionError(f"{method}任务ID不一致")
        graph_based = method not in {"category_nearest_current_detection", "clip_current_crop",
                                     "vlmaps_style_sparse_clip_grid"}
        for line, row, source in evidence:
            canonical = old_rows[key(row)]
            for field in ("positive", "query_text", "query_type", "acceptable_instances"):
                if row[field] != canonical[field]:
                    raise AssertionError("查询或GT被改写")
            raw_covered = row.get("raw_node_target_covered")
            ranked_covered = row.get("ranked_candidate_target_covered")
            if graph_based and row["positive"] and raw_covered is None:
                raw_covered = canonical["raw_node_target_covered"]
                association = associations[key(row)[:2]]
                ranked_covered = any(association.get(c) in row["acceptable_instances"]
                                     for c in row["candidate_ids"])
            normalized.append({
                "method": method, "task_id": f"{row['scene']}/f{row['frame_idx']}/q{row['query_index']}",
                **{k: row[k] for k in ("scene", "frame_idx", "query_index", "query_text", "query_type",
                                       "positive", "acceptable_instances", "selected", "evaluation_identity",
                                       "selected_correct", "false_selection_on_no_match")},
                "raw_node_target_covered": raw_covered,
                "ranked_candidate_target_covered": ranked_covered,
                "source_path": str(source), "source_line": line,
                "source_sha256": fingerprints[str(source)],
            })
        rr = normalized[-1115:]
        positives = [r for r in rr if r["positive"]]
        negatives = [r for r in rr if not r["positive"]]
        correct = sum(r["selected_correct"] is True for r in positives)
        false = sum(r["selected"] for r in negatives)
        rejected = sum(not r["selected"] for r in rr)
        unknown = sum(r["selected"] and r["evaluation_identity"] is None for r in rr)
        pos_selected = sum(r["selected"] for r in positives)
        known_wrong = sum(r["selected"] and not r["selected_correct"]
                          and r["evaluation_identity"] is not None for r in positives)
        pos_unknown = sum(r["selected"] and r["evaluation_identity"] is None for r in positives)
        raw_cover = sum(r["raw_node_target_covered"] is True for r in positives) if graph_based else None
        rank_cover = sum(r["ranked_candidate_target_covered"] is True for r in positives) if graph_based else None
        stats = (transfer_report if method in transfer_report["methods"] else relation_report)["methods"][method]
        expected_false = stats["no_match_false_selection"]
        if isinstance(expected_false, dict):
            expected_false = expected_false["numerator"]
        if (correct, false, rejected, unknown) != (stats["positive_selected_correct"], expected_false,
                                                  stats["no_selection"], stats["unknown_identity"]):
            raise AssertionError(f"{method}未复现冻结报告")
        summaries.append({
            "method": method, "label_zh": names[method],
            "map_or_input": "R1_full" if method.startswith("R1_") else
                            "old_one_to_one_depth_quality" if graph_based else "method_specific_frontend_or_grid",
            "tasks": 1115, "positive_denominator": 733, "negative_denominator": 382,
            "positive_correct": correct, "positive_correct_pct": percent(correct, 733),
            "negative_false_selection": false, "negative_false_selection_pct": percent(false, 382),
            "rejected": rejected, "rejected_pct": percent(rejected, 1115),
            "positive_rejected": 733 - pos_selected, "negative_rejected": 382 - false,
            "positive_selected": pos_selected, "positive_known_wrong": known_wrong,
            "positive_unknown": pos_unknown, "all_selected_unknown": unknown,
            "conditional_correct_among_positive_selected_pct": percent(correct, pos_selected),
            "raw_node_target_covered": raw_cover, "raw_node_target_covered_pct": percent(raw_cover, 733),
            "ranked_candidate_target_covered": rank_cover,
            "ranked_candidate_target_covered_pct": percent(rank_cover, 733),
            "coverage_note": "3m既有匹配下覆盖；排序候选指本方法现有过滤后候选" if graph_based else
                             "NA：原逐条记录未保存完整候选池，不能套用旧图候选覆盖",
            "comparison_scope": "固定地图比较规则" if graph_based and not method.startswith("R1_") else
                                "固定选择器比较地图版本" if method.startswith("R1_") else
                                "不同表示与关联协议的历史系统基线，非同图选择器消融",
            "source_path": str(evidence[0][2]),
            "vehicle_authorization_evaluated": False, "arrival_evaluated": False,
        })
    jsonl(args.run_dir / "selection_source_records.jsonl", normalized)
    table(args.report_dir / "target_selection_complete.csv", summaries)
    dump(args.report_dir / "target_selection_complete.json", {
        "split": "38验证调试场景/11log，非未见测试", "rows": summaries,
        "pairwise_R1_minus_old": transfer_report["paired_differences"],
        "pairwise_relation_fix_minus_old": relation_report["paired_differences"],
        "scope": "单时刻查询累积地图的离线受控目标检索；无本组Qwen调用、无颜色、无连续锁定或到达",
        "candidate_missing_is_failure_in_primary_denominator": True,
        "source_records": str(args.run_dir / "selection_source_records.jsonl"),
    })
    workpoints = read(WORKPOINTS / "report.json")
    table(args.report_dir / "shared_rejection_workpoints.csv", workpoints["all_shared_workpoints"])
    dump(args.report_dir / "shared_rejection_workpoints.json", workpoints)

    chains = {}
    chain_records = []
    for line, r in rows(GOALS / "queries.jsonl"):
        if key(r) not in old_rows or r["positive"] != old_rows[key(r)]["positive"]:
            raise AssertionError("接近规划任务与检索任务无法逐ID对应")
        selected = r["inference"].get("selected") or {}
        audit = selected.get("audit") or {}
        goal = selected.get("goal") or {}
        clearance_ok = goal.get("clearance_m") is not None and goal["clearance_m"] >= 0.5
        endpoint = r["endpoint_2m_from_predicted_bbox"] is True
        correct = r["positive"] and r["selected_correct"] is True
        approach = correct and endpoint and clearance_ok
        point_route = approach and r["route_generated"] is True and r["full_road_legal"] is True
        body = point_route and r["static_body_pose_legal"] is True
        bounded = body and audit.get("curvature_exceeds_vehicle_bound") is False
        chain_records.append({
            "method": r["method"], "task_id": f"{r['scene']}/f{r['frame_idx']}/q{r['query_index']}",
            "positive": r["positive"], "correct_selected": bool(correct),
            "correct_and_predicted_approach_conditions": bool(approach),
            "correct_approach_point_road_route": bool(point_route),
            "correct_approach_point_road_static_body_route": bool(body),
            "above_plus_curvature_bound": bool(bounded),
            "negative_false_selection": r["false_selection_on_no_match"] is True,
            "route_generated": r["route_generated"], "static_body_pose_legal": r["static_body_pose_legal"],
            "source_path": str(GOALS / "queries.jsonl"), "source_line": line,
        })
        chains.setdefault(r["method"], []).append(chain_records[-1])
    chain_table = []
    for method, rr in chains.items():
        if len(rr) != 1115 or len({r["task_id"] for r in rr}) != 1115:
            raise AssertionError("必须保留接近规划的完整1115任务")
        correct = sum(r["correct_selected"] for r in rr)
        approach = sum(r["correct_and_predicted_approach_conditions"] for r in rr)
        road = sum(r["correct_approach_point_road_route"] for r in rr)
        body = sum(r["correct_approach_point_road_static_body_route"] for r in rr)
        bounded = sum(r["above_plus_curvature_bound"] for r in rr)
        chain_table.append({
            "method": method, "map": "old_one_to_one_depth_quality", "tasks": 1115, "positive": 733,
            "negative": 382, "positive_correct": correct,
            "correct_and_predicted_approach_conditions": approach,
            "correct_approach_point_road_route": road, "point_road_chain_pct_of_all_positive": percent(road, 733),
            "correct_approach_point_road_static_body_route": body,
            "static_body_chain_pct_of_all_positive": percent(body, 733),
            "above_plus_curvature_bound": bounded,
            "approach_given_correct_pct": percent(approach, correct),
            "point_road_given_correct_approach_pct": percent(road, approach),
            "static_body_given_point_road_chain_pct": percent(body, road),
            "negative_false_selection": sum(r["negative_false_selection"] for r in rr),
            "all_generated_routes": sum(r["route_generated"] is True for r in rr),
            "all_static_body_passes": sum(r["static_body_pose_legal"] is True for r in rr),
            "continuous_lock": None, "continuous_body_sweep": None, "physical_arrival": None,
            "source_path": str(GOALS / "queries.jsonl"),
            "status": "历史逐条ID交集；接近点仅相对预测bbox/离散障碍，无真人停车区域GT",
        })
    chain_table.append({**{k: None for k in chain_table[0]}, "method": "R1_map_original_rule",
                        "map": "R1_full", "tasks": 1115, "positive": 733, "negative": 382,
                        "positive_correct": 81, "source_path": str(TRANSFER / "queries.jsonl"),
                        "status": "待补：R1没有对应接近点和路线记录；禁止借用旧图路线"})
    table(args.report_dir / "same_task_navigation_chain.csv", chain_table)
    dump(args.report_dir / "same_task_navigation_chain.json", {
        "rows": chain_table, "same_task_id_join": True, "minimum_clearance_m": 0.5,
        "minimum_predicted_bbox_distance_m": 2.0,
        "strict_success_is_not_actual_navigation_success": True,
        "not_evaluated": ["真人合法停车区域", "持续正确锁定", "连续车身扫掠", "控制器跟踪", "真实到达"],
        "evidence": str(args.run_dir / "same_task_chain_records.jsonl"),
    })
    jsonl(args.run_dir / "same_task_chain_records.jsonl", chain_records)

    graphs = dict((r["frame_idx"], (line, r)) for line, r in rows(EPISODE / "causal_graphs.jsonl"))
    r1_frames = dict((r["frame_idx"], (line, r)) for line, r in rows(R1_FRAME))
    inference = []
    pending = []
    for idx in range(40):
        source_line, source, source_path = old_frames[idx]
        graph_line, graph = graphs[idx]
        r1_line, r1 = r1_frames[idx]
        if not source["camera_timestamp_s"] == graph["camera_timestamp_s"] == r1["camera_timestamp_s"]:
            raise AssertionError("episode时间未对齐")
        image = Path(source["rgb"])
        if sha(image) != graph["rgb_sha256"]:
            raise AssertionError("审核图像改变")
        for node in r1["predictions"]["R1"]:
            if node.get("last_seen_frame", idx) > idx or node.get("timestamp_s", 0) > source["camera_timestamp_s"] + 1e-6:
                raise AssertionError("episode输入包含未来观测")
        inference.append({
            "episode_id": "scene0012_single_instruction_pilot", "scene": "scene-0012", "frame_idx": idx,
            "camera_timestamp_s": source["camera_timestamp_s"], "relative_s": graph["relative_s"],
            "rgb_path": str(image), "rgb_sha256": graph["rgb_sha256"],
            "camera_pose": source["camera_pose"], "intrinsic": source["intrinsic"],
            "map_version": "R1_full", "predicted_nodes": r1["predictions"]["R1"],
            "input_provenance": {"R1_path": str(R1_FRAME), "R1_line": r1_line,
                                 "R1_sha256": fingerprints[str(R1_FRAME)],
                                 "pose_source_path": str(source_path), "pose_source_line": source_line,
                                 "review_graph_path": str(EPISODE / "causal_graphs.jsonl"),
                                 "review_graph_line": graph_line,
                                 "review_graph_is_R1": False},
        })
        pending.append({
            "episode_id": "scene0012_single_instruction_pilot", "frame_idx": idx,
            "human_reviewed": False, "reviewer": None, "review_timestamp": None,
            "target_visible": None, "target_absent": None, "target_ambiguous": None,
            "target_image_box_or_click": None, "human_target_identity": None,
            "occlusion_state": None, "reference_identity": None, "color_verified": None,
            "evaluation_only_official_match": None,
        })
    jsonl(args.input_dir / "inference_frames.jsonl", inference)
    jsonl(args.input_dir / "evaluation_labels_pending.jsonl", pending)
    dump(args.input_dir / "task_pending_review.json", {
        "episode_id": "scene0012_single_instruction_pilot", "scene": "scene-0012",
        "instruction": None, "issue_frame_idx": None, "human_target_identity": None,
        "uses_color": None, "color_review_complete": False, "human_reviewed": False,
        "reviewer": None, "review_timestamp": None, "target_identity_audit_complete": False,
        "source_review_package": str(EPISODE / "index.html"),
        "note": "一次指令跨后续关键帧保持；空值表示尚无真人标注，不能由预测图回填标签",
    })
    protocol = {
        "protocol_version": "paper_nav_20261123_v1", "episode_id": "scene0012_single_instruction_pilot",
        "status": "blocked_human_annotation", "ready_to_execute": False,
        "frames": 40, "duration_s": inference[-1]["relative_s"],
        "instruction_count": 1, "issue_frame_idx": None,
        "split": "已使用验证场景的工作流pilot，不是独立测试",
        "inference_input": str(args.input_dir / "inference_frames.jsonl"),
        "task": str(args.input_dir / "task_pending_review.json"),
        "evaluation_only_labels": str(args.input_dir / "evaluation_labels_pending.jsonl"),
        "causal_rule": "时刻t只消费一条指令、RGB/位姿和<=t预测图；评估标签不能输入解析/排序/状态机",
        "review_graph_boundary": "原审核包图为frontend_v2_evidence；本新清单另引用R1，不能称原审核图已是R1",
        "not_completed": ["真人自然指令", "指令发出时刻", "目标/参照身份", "逐时刻可见/遮挡/歧义",
                          "白色等颜色属性复核", "状态机执行", "锁定/接近点/路线/到达评价"],
        "future_output_per_frame": ["task_id", "time", "raw_model_response", "parse_status", "fallback_used",
                                    "proposal_id", "selected_id", "authorized", "rejection_reason",
                                    "lock_id", "state", "goal", "path", "geometry_audit", "timing"],
        "metric_protocol": {
            "positive_episode_denominator": "全部经真人审核有明确目标的指令episode；不能按候选覆盖筛选",
            "negative_episode_denominator": "全部经真人审核整段无匹配目标的指令episode",
            "negative_false_selection": "每负episode是否曾误选，并报告误选时长/已审核可评价时长",
            "first_localization_delay": "从已审核首次可定位时刻到首次正确实例选择；未找到右删失，报告找到率与延迟，不删除失败",
            "sustained_lock": "从首次正确选择后，在已审核可评价帧的正确锁定时长比例；遮挡后再现另列；状态机需预先冻结",
            "identity_switches": "只在可评价的连续锁定轨迹计真实目标身份切换；预测node换号与目标身份变更分列",
            "offline_chain": "同episode同目标依次列正确定位/持续锁定/合格接近点/合格路线的全episode交集及阶段条件成功率",
            "actual_arrival": "NA：nuScenes回放没有车辆按指令执行；不能填0或模拟到达",
            "unfrozen_parameters": ["锁定进入/退出规则", "允许短遮挡时长", "可定位标注规则", "合格接近点/路线的完整条件"],
        },
    }
    dump(args.input_dir / "episode_protocol.json", protocol)
    dump(args.report_dir / "episode_protocol_status.json", protocol)
    dump(args.run_dir / "aggregation_summary.json", {
        "selection_methods": len(summaries), "selection_records": len(normalized),
        "chain_methods_with_real_routes": len(chains), "chain_records": len(chain_records),
        "episode_frames": len(inference), "human_annotations_created": 0,
        "episode_ready": False, "elapsed_s": time.monotonic() - started,
        "model_calls": 0, "planner_calls": 0, "new_experiment_calls": 0,
        "tables": str(args.report_dir), "episode_inputs": str(args.input_dir),
    })
    if any(sha(path) != expected for path, expected in fingerprints.items()):
        raise AssertionError("汇总过程中历史输入发生变化")
    snapshot = args.run_dir / "source_snapshot/scripts"
    snapshot.mkdir(parents=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    print(json.dumps({"selection": summaries[:2], "chain": chain_table,
                      "episode_status": protocol["status"], "elapsed_s": time.monotonic() - started},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
