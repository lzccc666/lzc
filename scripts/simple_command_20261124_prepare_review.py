"""生成普通指令v3审核包：去除模板完成子句，原文只保留在独立来源记录。"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

from jsonschema import Draft202012Validator
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


PACKAGE = "multiview_task_20261124_simple_candidates_v3"
RUN_ID = "simple_command_review_20261124_v3"
ROOT = PROJECT_ROOT
OLD = ROOT / "dataset/qwen/multiview_task_20261123_intent_candidates_v1"
V2 = ROOT / "dataset/qwen/multiview_task_20261124_openloop_candidates_v2"
TEMPLATE = ROOT / "conceptgraph/annotation/simple_command_20261124_review.html"
REPORT = ROOT / "outputs/reports/command_review_simplify_20261124"
SIMPLIFIED = {
    "PO07__p1": "请在允许停车且空闲的路边停车。",
    "PO07__p2": "找允许停靠、又没被占用的路边位置停车。",
    "PO07__p3": "麻烦靠边停到允许停车的空闲位置。",
    "PK07__p1": "停到左侧最近的空闲车位。",
    "PK07__p2": "请找左边最近的空车位停好。",
    "PK07__p3": "麻烦在左侧最近空闲停车位泊车。",
    "FV01__p1": "请跟随本车道前车。",
    "FV01__p2": "跟着同车道的前车行驶。",
    "FV01__p3": "麻烦保持跟随本车道前方车辆。",
    "FV04__p1": "请跟随本车道白色前车。",
    "FV04__p2": "跟着同车道前方的白色车走。",
    "FV04__p3": "麻烦跟随本车道那辆白色前车。",
    "FV08__p1": "跟着那辆车，先确认一下我指的是哪一辆。",
    "FV08__p2": "请跟随我说的那辆车，我还没指明是哪辆，先问清楚。",
    "FV08__p3": "麻烦跟随那台车，目标车辆先向我确认。",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    begun = time.monotonic()
    folder = ROOT / "dataset/qwen" / PACKAGE
    if folder.exists():
        raise FileExistsError(folder)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("磁盘不足20GiB，停止生成")
    protected = [p for base in (OLD, V2) for p in base.iterdir() if p.is_file()]
    protected += [ROOT / p for p in (
        "AGENTS.md", "MEMORY.md", "PAPER_MEMORY.md", "paper/当前版_v4/sources/论文正文.md",
        "conceptgraph/annotation/multiview_task_20261123_intent_review.html",
        "conceptgraph/annotation/openloop_20261124_intent_review.html",
        "scripts/multiview_task_20261123_prepare_qwen_candidates.py",
        "scripts/openloop_20261124_prepare_intent_review.py",
        "conceptgraph/navigation/semantic_tasks_20261123.py")]
    before = {str(p): sha(p) for p in protected}
    run = experiment_directory("qwen", "preparation", RUN_ID, create=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(run / "registration.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_budget_s": 900, "additional_bytes_budget": 64 * 1024**2,
        "minimum_free_bytes": 20 * 1024**3, "model_calls": 0, "training_calls": 0,
        "protected_before": before})
    originals = [json.loads(s) for s in (OLD / "candidates.jsonl").read_text().splitlines()]
    assert len(originals) == 72
    schema = json.loads((OLD / "language_intent.schema.json").read_text())
    schema["title"] = "SimpleNavigationCommandIntentV3"
    schema["required"].remove("termination_request")
    del schema["properties"]["termination_request"]
    schema["properties"]["requested_constraints"]["properties"]["duration_s"] = {
        "type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 600,
        "description": "仅当原句明确请求跟随时长时填写；普通指令不强制给时长。"}
    schema["description"] = "普通导航指令的动作、目标、明确请求及澄清；不含任务完成合同。"
    schema["$comment"] = "语言审核只核说法与理解；完成条件由程序处理。无终止模式、执行授权、环境真值或轨迹通过标签。"
    validator = Draft202012Validator(schema)
    rows, history = [], []
    for original in originals:
        instruction = SIMPLIFIED.get(original["sample_id"], original["instruction"])
        assert not re.search(r"结束|便完成|便算任务完成|叫停任务", instruction)
        intent = deepcopy(original["intent_draft"])
        termination = intent.pop("termination_request")
        if termination["duration_s"] is not None:
            intent["requested_constraints"]["duration_s"] = termination["duration_s"]
        validator.validate(intent)
        sample_id = original["sample_id"] + "__v3"
        rows.append({"sample_id": sample_id, "display_id": original["sample_id"],
            "paraphrase_family_id": original["paraphrase_family_id"],
            "paraphrase_index": int(original["sample_id"].rsplit("__p", 1)[1]),
            "package_id": PACKAGE, "instruction": instruction, "instruction_sha256": text_sha(instruction),
            "intent_draft": intent, "source_history_id": sample_id,
            "source_kind": "programmatic_template_candidate_simplified_v3",
            "human_review_status": "pending", "human_review": {"reviewer": None, "reviewed_at": None, "decision": None, "notes": None},
            "training_eligible": False, "scene": None, "log_token": None, "episode_id": None,
            "split": "unassigned", "grounding_status": "unbound_unknown"})
        history.append({"sample_id": sample_id, "original_package_id": original["package_id"],
            "original_sample_id": original["sample_id"], "original_instruction": original["instruction"],
            "original_instruction_sha256": original["instruction_sha256"],
            "original_intent_draft": original["intent_draft"], "original_source": original["source"],
            "instruction_changed": instruction != original["instruction"],
            "change": "remove_completion_clause_preserve_ordinary_action" if instruction != original["instruction"] else "text_unchanged_new_schema_and_versioned_ID",
            "human_review_input": False, "training_label": False})
    assert len(rows) == len({r["sample_id"] for r in rows}) == 72
    assert sum(h["instruction_changed"] for h in history) == 15
    folder.mkdir()
    for name, objects in (("candidates.jsonl", rows), ("source_history.jsonl", history)):
        (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in objects))
    dump(folder / "simple_intent.schema.json", schema)
    shutil.copy2(OLD / "group_split_policy.json", folder / "group_split_policy.json")
    with (folder / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "instruction", "action", "target_description", "requested_constraints", "human_review_status", "training_eligible"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"sample_id": r["sample_id"], "instruction": r["instruction"], "action": r["intent_draft"]["action"],
                "target_description": r["intent_draft"]["target_description"],
                "requested_constraints": json.dumps(r["intent_draft"]["requested_constraints"], ensure_ascii=False),
                "human_review_status": "pending", "training_eligible": False})
    manifest = {"package_id": PACKAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 72, "families": 24, "task_counts": {a: sum(r["intent_draft"]["action"] == a for r in rows) for a in ("pull_over", "park_nearest", "follow_vehicle")},
        "source_package_id": "multiview_task_20261123_intent_candidates_v1",
        "source_candidates_sha256": sha(OLD / "candidates.jsonl"), "candidates_sha256": sha(folder / "candidates.jsonl"),
        "source_history_sha256": sha(folder / "source_history.jsonl"),
        "intent_schema_sha256": sha(folder / "simple_intent.schema.json"),
        "UI_template_sha256": sha(TEMPLATE), "source_script_sha256": sha(Path(__file__)),
        "changed_instruction_rows": 15, "unchanged_instruction_text_rows": 57, "versioned_ID_rows": 72,
        "human_reviewed_rows": 0, "training_eligible_rows": 0, "training_export_enabled": False,
        "model_calls": 0, "GPU_calls": 0, "trajectory_evaluation_calls": 0,
        "source_history_is_review_or_training_input": False,
        "old_review_import": "rejected_no_automatic_approval_migration",
        "completion_conditions": "program_responsibility_not_language_review_input",
        "data_split": "unassigned", "episode_bindings": 0}
    dump(folder / "manifest.json", manifest)
    bundle = {"manifest": manifest, "manifest_sha256": sha(folder / "manifest.json"), "schema": schema, "rows": rows}
    (folder / "index.html").write_text(TEMPLATE.read_text().replace("__BUNDLE_JSON__", json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c")))
    (folder / "审核说明.txt").write_text("打开index.html，只看普通指令和系统理解是否一致。完成条件由程序处理。\n选择一致、需修改、待审或不采用，可改说法和写备注；逐条保存，最后导出JSON发回聊天。默认每组首条24条，完整候选72条；单条通过不自动批准同组其他说法。\n旧版本意见原样发回，不会自动转成新版通过。source_history.jsonl只供来源追溯，不是本页审核内容或训练答案。\n")
    with zipfile.ZipFile(folder / "普通导航指令_72条简版审核包.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix != ".zip":
                archive.write(p, arcname=p.name)
    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    for p in (Path(__file__), TEMPLATE, ROOT / "conceptgraph/project_paths.py"):
        shutil.copy2(p, snapshot / p.name)
    assert all(sha(Path(p)) == value for p, value in before.items())
    check = {"passed": True, "rows": 72, "changed_text_rows": 15, "unchanged_text_rows": 57,
        "new_version_IDs": 72, "schema_valid_rows": 72, "completion_clause_rows": 0,
        "review_bundle_contains_source_history": False, "pending_rows": 72, "training_eligible_rows": 0,
        "human_labels_created": 0, "models_training_trajectory_driving_calls": 0,
        "protected_count": len(before), "protected_SHA_unchanged": True, "protected_SHA": before,
        "package_SHA": {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
        "wall_s": time.monotonic() - begun,
        "new_bytes": sum(p.stat().st_size for base in (folder, run) for p in base.rglob("*") if p.is_file()),
        "free_bytes": shutil.disk_usage(ROOT).free, "package": str(folder), "run": str(run)}
    assert check["new_bytes"] < 64 * 1024**2 and check["free_bytes"] >= 20 * 1024**3
    dump(run / "preparation_verification.json", check)
    dump(REPORT / "preparation_verification.json", check)
    print(json.dumps({"package": str(folder), "run": str(run), "changed_rows": 15}, ensure_ascii=False))


if __name__ == "__main__":
    main()
