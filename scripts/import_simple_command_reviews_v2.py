"""Freeze and prepare a new, nontraining import of submitted command-review JSON."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from conceptgraph.annotation.command_review_import_v2 import (
    derive_records, file_sha256, load_json, load_source_package, rejection_checks, sha256_bytes,
)
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--import-id", required=True)
    parser.add_argument("--chat-clarification", type=Path)
    parser.add_argument("--expected-clarification-sha256")
    args = parser.parse_args()
    started = time.monotonic()
    free_before = shutil.disk_usage(PROJECT_ROOT).free
    if free_before < 20 * 1024**3:
        raise RuntimeError("磁盘空闲低于20GiB，停止新增数据")
    source_path, review_path = args.source_package.resolve(), args.review.resolve()
    raw = review_path.read_bytes()
    raw_sha = sha256_bytes(raw)
    if raw_sha != args.expected_review_sha256:
        raise ValueError("用户原件SHA与登记不符")
    source = load_source_package(source_path)
    payload = load_json(raw)
    clarification = None
    if args.chat_clarification:
        if not args.expected_clarification_sha256 or file_sha256(args.chat_clarification) != args.expected_clarification_sha256:
            raise ValueError("补充更正SHA与登记不符")
        clarification = load_json(args.chat_clarification.read_bytes())
    derived = derive_records(payload, source, raw_sha, clarification)
    negative_checks = rejection_checks(payload, source)
    protected = set(source_path.rglob("*"))
    for package_name in ["multiview_task_20261123_intent_candidates_v1", "multiview_task_20261124_openloop_candidates_v2"]:
        protected.update((PROJECT_ROOT / "dataset/qwen" / package_name).rglob("*"))
    protected.update(PROJECT_ROOT / name for name in ["AGENTS.md", "MEMORY.md", "PAPER_MEMORY.md", "paper/当前版_v4/sources/论文正文.md"])
    protected.add(review_path)
    if args.chat_clarification:
        protected.add(args.chat_clarification.resolve())
    if args.import_id.endswith("_v2"):
        protected.update((PROJECT_ROOT / "dataset/qwen/command_review_import_20261124_v1").rglob("*"))
        protected.update((PROJECT_ROOT / "outputs/results/qwen/preparation/iterations/command_review_import_20261124_v1").rglob("*"))
        protected.update(PROJECT_ROOT / path for path in ["conceptgraph/annotation/command_review_import.py", "scripts/import_simple_command_reviews.py"])
        protected.add(PROJECT_ROOT / "outputs/reports/command_review_import_20261124/import_verification.json")
    protected_sha = {str(path): file_sha256(path) for path in sorted(protected) if path.is_file()}
    run = experiment_directory("qwen", "preparation", args.import_id, create=True)
    data = PROJECT_ROOT / "dataset/qwen" / args.import_id
    data.mkdir(exist_ok=False)
    report = PROJECT_ROOT / "outputs/reports/command_review_import_20261124"
    report.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(), PROJECT_ROOT / "conceptgraph/annotation/command_review_import_v2.py", PROJECT_ROOT / "conceptgraph/project_paths.py"]
    source_sha = {str(path): file_sha256(path) for path in sources}
    for path in sources:
        destination = run / "source_snapshot" / path.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    input_sha = {str(path): file_sha256(path) for path in sorted(source_path.rglob("*")) if path.is_file()}
    input_sha[str(review_path)] = raw_sha
    if args.chat_clarification:
        input_sha[str(args.chat_clarification.resolve())] = file_sha256(args.chat_clarification)
    registration = dict(run=str(run), input_sha256=input_sha, source_sha256=source_sha,
                        random_seed=None, random_operations=False, budget=dict(minutes=15, new_bytes=64 * 1024**2),
                        identity_verified=False, training_export_enabled=False, model_calls=0, GPU_calls=0,
                        training_calls=0, planning_calls=0, git_mutations=0)
    write_json(run / "registration.json", registration)
    (data / "source_review.json").write_bytes(raw)
    if args.chat_clarification:
        shutil.copyfile(args.chat_clarification, data / "chat_confirmation.json")
    write_jsonl(data / "candidate_review_ledger.jsonl", derived["ledger"])
    write_jsonl(data / "approved_language_intents.jsonl", derived["approved"])
    write_json(data / "split_group_manifest.json", dict(
        split="unassigned", grouping_rule="connected shared family, identical intent, or bound scene/log/episode",
        groups=derived["groups"], assigned_training_rows=0, assigned_validation_rows=0, assigned_test_rows=0,
        no_automatic_family_approval=True, scene_binding_required_before_grounded_evaluation=True))
    manifest = dict(
        import_id=args.import_id, source_package_id=payload["package_id"],
        received_via="user_submitted_chat_attachment", review_source_sha256=raw_sha,
        source_manifest_sha256=payload["manifest_sha256"], source_schema_sha256=payload["intent_schema_sha256"],
        source_candidates_sha256=source["manifest"]["candidates_sha256"],
        candidate_rows=len(derived["ledger"]), review_events=len(payload["review_events"]),
        approved_language_intent_rows=len(derived["approved"]), counts=derived["counts"],
        raw_counts=derived["raw_counts"], chat_confirmation_rows=int(clarification is not None),
        task_counts=derived["task_counts"], identity_verified=payload["identity_verified"],
        training_export_enabled=False, training_eligible_rows=0, split="unassigned", scene_bindings=0,
        human_reviewed_template_candidates=True, natural_user_original_instruction_rows=0,
        trajectory_evaluation_status="not_evaluated", rejection_is_model_failure=False,
        review_scope="submitted sample only; no propagated approval or permanent capability exclusions",
        files_sha256={p.name: file_sha256(p) for p in sorted(data.iterdir())},
    )
    write_json(data / "manifest.json", manifest)
    assert file_sha256(data / "source_review.json") == raw_sha
    assert len(derived["ledger"]) == len(source["rows"])
    assert all(not row["training_eligible"] for row in derived["ledger"])
    approved_ids = {row["sample_id"] for row in derived["approved"]}
    assert all(row["sample_id"] not in approved_ids for row in derived["ledger"] if row["review_event"] is None)
    groups_by_id = {sample_id: group["split_group_id"] for group in derived["groups"] for sample_id in group["sample_ids"]}
    assert groups_by_id["PK03__p1__v3"] == groups_by_id["PK07__p1__v3"]
    previous_path = PROJECT_ROOT / "dataset/qwen/command_review_import_20261124_v1/candidate_review_ledger.jsonl"
    previous = {row["sample_id"]: row for row in [load_json(line.encode("utf-8")) for line in previous_path.read_text().splitlines()]}
    unique_delta = []
    for row in derived["ledger"]:
        old = previous[row["sample_id"]]
        old.update(raw_decision=old["review_decision"], effective_decision=old["review_decision"], chat_confirmation=None)
        if row != old:
            unique_delta.append(row["sample_id"])
    assert unique_delta == [clarification["sample_id"]]
    assert derived["counts"] == {"approve": 16, "unreviewed": 48, "reject": 8}
    assert len(derived["approved"]) == 16
    assert all(file_sha256(Path(path)) == old_sha for path, old_sha in protected_sha.items())
    assert all(file_sha256(Path(path)) == old_sha for path, old_sha in source_sha.items())
    verification = dict(
        passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), data=str(data), run=str(run),
        counts=derived["counts"], task_counts=derived["task_counts"], total_candidates=len(derived["ledger"]),
        raw_counts=derived["raw_counts"], chat_confirmation_rows=int(clarification is not None),
        only_changed_sample_ids=unique_delta, other_candidate_content_unchanged=71,
        imported_unique_events=len(payload["review_events"]), approved_language_intent_rows=len(derived["approved"]),
        unchanged_original_review_bytes=True, no_family_approval_propagation=True,
        PK03_PK07_same_split_group=True, negative_checks=negative_checks,
        protected_count=len(protected_sha), protected_SHA_unchanged=True,
        protected_sha256=protected_sha, source_SHA_unchanged=True,
        data_sha256={p.name: file_sha256(p) for p in sorted(data.iterdir())},
        no_training_or_model_calls=True, no_new_task_or_trajectory_scores=True,
        wall_s=time.monotonic() - started, free_bytes=shutil.disk_usage(PROJECT_ROOT).free,
        data_and_run_bytes=sum(p.stat().st_size for folder in [data, run] for p in folder.rglob("*") if p.is_file()),
    )
    write_json(run / "import_verification.json", verification)
    write_json(report / "import_verification_v2.json", verification)
    print(json.dumps({k: verification[k] for k in ["passed", "data", "run", "counts", "task_counts", "approved_language_intent_rows", "wall_s", "data_and_run_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
