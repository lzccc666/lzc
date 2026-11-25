"""从原72条候选生成独立开环审核包；不迁移批准、不评价轨迹、不调用模型。"""
from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from jsonschema import Draft202012Validator

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


PACKAGE = "multiview_task_20261124_openloop_candidates_v2"
RUN_ID = "openloop_review_fix_20261124_v2"
OLD_PACKAGE = PROJECT_ROOT / "dataset/qwen/multiview_task_20261123_intent_candidates_v1"
TEMPLATE = PROJECT_ROOT / "conceptgraph/annotation/openloop_20261124_intent_review.html"
REPORT = PROJECT_ROOT / "outputs/reports/openloop_review_fix_20261124"
PROTOCOL = {
    "protocol_id": "openloop_trajectory_intent_review_20261124_v2",
    "evaluation_mode": "open_loop",
    "evaluation_object": "predicted_trajectory",
    "review_layer": "language_intent_only",
    "requires_actual_execution": False,
    "requires_actual_stop_position": False,
    "requires_actual_speed": False,
    "requires_actual_follow_duration": False,
    "text_approval_is_trajectory_success": False,
    "trajectory_evaluation_status": "not_evaluated",
    "actual_execution_status": "not_evaluated_open_loop",
    "actual_arrival_result": None,
    "parking_completion": "predicted_terminal_state_satisfies_task_goal",
    "following_completion": "predicted_trajectory_satisfies_task_within_preregistered_finite_horizon",
    "requested_follow_duration": "language_request_not_actual_elapsed_wait",
    "user_end": "language_request_not_review_or_prediction_wait_gate",
    "speed_evidence": "predicted_velocity_or_timestamped_trajectory_if_required_else_unknown",
    "spatial_path_only": "does_not_verify_speed_or_stopped_state",
    "unknown_required_conditions": "unknown_not_automatic_success",
    "trajectory_correctness": "correct_task_target_and_geometry_and_available_required_constraints",
    "nonempty_path_is_sufficient": False,
    "recorded_ego_future_is_automatic_task_ground_truth": False,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    start = time.monotonic()
    if shutil.disk_usage(PROJECT_ROOT).free < 20 * 1024**3:
        raise RuntimeError("磁盘不足20GiB，停止生成")
    destination = PROJECT_ROOT / "dataset/qwen" / PACKAGE
    if destination.exists():
        raise FileExistsError(destination)
    protected = list(OLD_PACKAGE.glob("*")) + [
        PROJECT_ROOT / "conceptgraph/annotation/multiview_task_20261123_intent_review.html",
        PROJECT_ROOT / "scripts/multiview_task_20261123_prepare_qwen_candidates.py",
        PROJECT_ROOT / "conceptgraph/navigation/semantic_tasks_20261123.py",
        PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "MEMORY.md", PROJECT_ROOT / "PAPER_MEMORY.md",
        PROJECT_ROOT / "paper/当前版_v4/sources/论文正文.md",
    ]
    protected_before = {str(p): sha(p) for p in protected if p.is_file()}
    run = experiment_directory("qwen", "preparation", RUN_ID, create=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(run / "registration.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "修复任务意图审核中的执行/预测混淆；无轨迹评价成绩",
        "wall_budget_s": 1200, "additional_bytes_budget": 128 * 1024**2,
        "minimum_free_bytes": 20 * 1024**3, "model_calls": 0, "GPU_calls": 0,
        "input_package": str(OLD_PACKAGE), "protected_before": protected_before,
    })
    original_rows = [json.loads(s) for s in (OLD_PACKAGE / "candidates.jsonl").read_text().splitlines()]
    assert len(original_rows) == 72
    assert all(r["human_review_status"] == "pending" and not r["training_eligible"] for r in original_rows)
    rows = deepcopy(original_rows)
    for row in rows:
        row["package_id"] = PACKAGE
        row["review_protocol_id"] = PROTOCOL["protocol_id"]
        row["source_package_id"] = "multiview_task_20261123_intent_candidates_v1"
    schema = json.loads((OLD_PACKAGE / "language_intent.schema.json").read_text())
    schema["title"] = "OpenLoopLanguageIntentReview20261124V2"
    schema["description"] = "指令请求的任务目标；本包仅开环预测轨迹语义，不要求实际执行。文字通过不等于轨迹正确。"
    schema["properties"]["termination_request"]["description"] = (
        "保留原语言意图模式及原句含义。停车模式指预测轨迹终端目标；跟随时长为语言请求；"
        "user_end不要求审核/预测等待真实用户结束。实际评价在预先固定的有限预测时域进行。"
        "时间/速度不可用时不声称速度或停稳已核验，不要求用户补实际遥测。"
    )
    schema["$comment"] += " 开环范围及独立轨迹评价状态由review_protocol.json固定；不修改原v1 schema。"
    validator = Draft202012Validator(schema)
    for original, row in zip(original_rows, rows):
        validator.validate(row["intent_draft"])
        assert row["instruction"] == original["instruction"]
        assert row["instruction_sha256"] == original["instruction_sha256"]
        assert row["sample_id"] == original["sample_id"]
        assert row["intent_draft"] == original["intent_draft"]
        assert all(row[k] is None for k in ("scene", "log_token", "episode_id"))
        assert row["human_review"] == original["human_review"]
    destination.mkdir()
    (destination / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    fields = ["sample_id", "instruction", "action", "termination_request", "review_protocol_id", "human_review_status", "training_eligible"]
    with (destination / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"sample_id": r["sample_id"], "instruction": r["instruction"],
                "action": r["intent_draft"]["action"],
                "termination_request": json.dumps(r["intent_draft"]["termination_request"], ensure_ascii=False),
                "review_protocol_id": r["review_protocol_id"], "human_review_status": "pending", "training_eligible": False})
    dump(destination / "language_intent.schema.json", schema)
    dump(destination / "review_protocol.json", PROTOCOL)
    shutil.copy2(OLD_PACKAGE / "group_split_policy.json", destination / "group_split_policy.json")
    manifest = {
        "package_id": PACKAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 72, "families": 24, "task_counts": {k: sum(r["intent_draft"]["action"] == k for r in rows) for k in ("pull_over", "park_nearest", "follow_vehicle")},
        "source_package_id": "multiview_task_20261123_intent_candidates_v1",
        "source_candidates_sha256": sha(OLD_PACKAGE / "candidates.jsonl"),
        "source_kind": "unchanged_programmatic_template_candidates_new_review_protocol",
        "candidates_sha256": sha(destination / "candidates.jsonl"),
        "intent_schema_sha256": sha(destination / "language_intent.schema.json"),
        "review_protocol_sha256": sha(destination / "review_protocol.json"),
        "source_script_sha256": sha(Path(__file__)), "UI_template_sha256": sha(TEMPLATE),
        "human_reviewed_rows": 0, "actual_training_rows": 0,
        "training_export_enabled": False, "data_split": "unassigned", "episode_bindings": 0,
        "model_calls": 0, "GPU_calls": 0, "trajectory_predictions": 0,
        "old_review_import": "rejected; no implicit migration or transferred approval",
        "intent_draft_policy": "same 72 original instructions/IDs/intent drafts; parking/following describe predicted-task goals under fixed open-loop protocol",
    }
    dump(destination / "manifest.json", manifest)
    bundle = {"manifest": manifest, "manifest_sha256": sha(destination / "manifest.json"),
        "schema": schema, "review_protocol": PROTOCOL, "rows": rows}
    encoded = json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c")
    template = TEMPLATE.read_text()
    assert template.count("__BUNDLE_JSON__") == 1
    (destination / "index.html").write_text(template.replace("__BUNDLE_JSON__", encoded))
    instructions = """打开index.html。默认每家族首条，共24条；72条均待审，单条保存不会批准其他改写。

本批是开环：只要求预测轨迹正确，不等实际停车、实际速度下降或实际跟满时长。
停车位置、航向和速度如有要求，检查预测轨迹及其末端；若只有空间路径，没有时间/速度输出，不声称速度或停稳已核。跟车按预先固定的有限预测时域核轨迹，指令里的10/30秒仍是意图请求，不是实际等待；“直到用户结束”也不阻塞本批预测评估。

本页只审核语言意图：指令想让预测轨迹做什么。无需你填写实际停车坐标、实际车速、车辆遥测或设计实验。动作/目标/条件有误写备注或改说法，无法确认就保留待审。文字“通过”不等于模型预测轨迹通过；后续轨迹评价另核目标、几何及所需且有依据的条件。非空路线不足以判正确，录制的原车未来轨迹不能自动当任意新指令的标准答案。

填写审核人代号、核含义，逐条保存，最后导出JSON发回聊天。保存仅在本浏览器；没有上传。新v2与旧v1缓存隔离，旧JSON不能直接导入v2，旧意见与原始文件保留。若已在v1做过审核，请将旧JSON原样发回，另行核对含义，不能自动把旧批准转成v2批准。

本包0条人工真值、0条训练可用、0次轨迹/模型评价。scene/log/episode未绑定、split未分配，原分组策略继续有效。不使用本包改写历史分数、原稿或CARLA执行监控。
"""
    (destination / "审核说明.txt").write_text(instructions)
    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), TEMPLATE, PROJECT_ROOT / "conceptgraph/project_paths.py"):
        shutil.copy2(source, snapshot / source.name)
    with zipfile.ZipFile(destination / "开环轨迹任务意图_72条待审核离线包.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, arcname=path.name)
    protected_after = {p: sha(Path(p)) for p in protected_before}
    assert protected_before == protected_after
    checks = {
        "passed": True, "package": str(destination), "run": str(run),
        "rows": 72, "same_original_instructions_IDs_intents": True,
        "schema_valid_rows": 72, "pending_rows": 72, "training_eligible_rows": 0,
        "new_human_labels": 0, "model_calls": 0, "GPU_calls": 0, "trajectory_scores": 0,
        "protected_count": len(protected_before), "protected_SHA_unchanged": protected_before == protected_after,
        "protected_SHA": protected_after,
        "package_SHA": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
        "wall_s": time.monotonic() - start,
        "new_bytes": sum(p.stat().st_size for parent in (destination, run) for p in parent.rglob("*") if p.is_file()),
        "free_bytes": shutil.disk_usage(PROJECT_ROOT).free,
    }
    assert checks["new_bytes"] < 128 * 1024**2 and checks["free_bytes"] >= 20 * 1024**3
    dump(run / "preparation_verification.json", checks)
    dump(REPORT / "preparation_verification.json", checks)
    print(json.dumps({"package": str(destination), "run": str(run), "verification": str(REPORT / "preparation_verification.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
