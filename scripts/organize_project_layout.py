"""整理输入数据并建立分类入口；保留原路径，逐文件核对 SHA-256。

默认只输出计划。--apply 实际执行，--verify 复核已有迁移。
不删除历史文件，不改写数据、模型、源码和实验报告。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "outputs/project_v2/thesis_completion_v1"
PUBLIC = OLD / "public_experiment_completion_20261115"
AUDIT = OLD / "project_organization_20261118"


def moves():
    preparation = PROJECT / "dataset/processed/preparation_history"
    rows = [
        (OLD / "data_manifest", preparation),
        (preparation / "development_causal_inputs_full_001",
         PROJECT / "dataset/processed/nuscenes/validation"),
        (preparation / "reserved_causal_inputs_full_001",
         PROJECT / "dataset/processed/nuscenes/test"),
        (OLD / "semantic_dataset", PROJECT / "dataset/qwen/structured_supervision_history"),
    ]
    for family, versions in (("official", (1, 2)), ("real", (1, 2, 3))):
        for version in versions:
            rows.append((PUBLIC / f"qwen_adapter_fair_comparison/{family}_graph_dataset_{version:03d}",
                         PROJECT / f"dataset/qwen/{family}_graph_v{version:03d}"))
    return rows


def links():
    bulk = Path("/root/autodl-fs/thesis_completion_v1_bulk/kitti_tracking_expanded_006")
    return {
        "dataset/raw/nuscenes_trainval": PROJECT / "dataset/nuscenes_trainval_raw",
        "dataset/raw/nuscenes_mini": PROJECT / "dataset/v1.0-mini",
        "dataset/raw/kitti_tracking_downloads": bulk / "official_small_archives",
        "dataset/raw/kitti_tracking_labels": bulk / "evaluation_only",
        "dataset/processed/kitti/tracking": bulk / "agent_inputs",
        "dataset/processed/kitti/manifest": PROJECT / "dataset/processed/preparation_history/kitti_tracking_expanded_006",
        "dataset/qwen/history_stage1_8": PROJECT / "outputs/qwen_finetune_dataset",
        "dataset/annotations/natural_navigation_submissions": PUBLIC / "annotation/human_generated_natural_submissions_071",
        "dataset/annotations/human_attributes": PUBLIC / "annotation/human_attribute_reference_003",
        "dataset/splits/historical": OLD / "splits",
        "outputs/results/nuscenes/validation/perception": OLD / "mapping_evaluation/causal_frontend_reuse_full_001",
        "outputs/results/nuscenes/validation/mapping": PUBLIC / "nuscenes_ablation/strict_full_001",
        "outputs/results/nuscenes/validation/ablations": PUBLIC / "nuscenes_ablation",
        "outputs/results/nuscenes/validation/statistics": PUBLIC / "nuscenes_ablation/statistics_001",
        "outputs/results/nuscenes/test/perception": PUBLIC / "main_experiments/sealed_public_058/frontend",
        "outputs/results/nuscenes/test/mapping": PUBLIC / "main_experiments/sealed_public_058/mapping",
        "outputs/results/nuscenes/test/evaluation": PUBLIC / "main_experiments/sealed_execution_075/evaluation",
        "outputs/results/nuscenes/test/requests": PUBLIC / "main_experiments/sealed_execution_075/requests",
        "outputs/results/kitti/perception": PUBLIC / "mapping_evaluation/kitti_gpu_frontend_full_001",
        "outputs/results/kitti/mapping": PUBLIC / "mapping_evaluation/kitti_mapping_full_001",
        "outputs/results/kitti/evaluation": PUBLIC / "statistics/kitti_expanded_full_001",
        "outputs/results/kitti/viewer": PUBLIC / "kitti_tracking_expanded",
        "outputs/results/qwen/training": PUBLIC / "qwen_adapter_fair_comparison/training",
        "outputs/results/qwen/development_evaluation": PUBLIC / "qwen_adapter_fair_comparison/real_graph_fair_gpu_001",
        "outputs/results/qwen/test_evaluation": PUBLIC / "main_experiments/sealed_execution_075/evaluation",
        "outputs/results/qwen/diagnostics": PUBLIC / "qwen_adapter_fair_comparison/real_graph_failure_taxonomy_001",
        "outputs/results/qwen/history": OLD / "training",
        "outputs/results/carla": PUBLIC / "carla_official",
        "outputs/results/comparisons": PUBLIC / "method_comparison",
        "outputs/results/tables": PUBLIC / "tables",
        "outputs/results/figures": PUBLIC / "figures",
        "outputs/results/reproduction": PUBLIC / "reproduction",
        "outputs/reports/organization": AUDIT,
        "outputs/reports/public_experiment": PUBLIC,
        "scripts/experiment_archive/public_20261115": PUBLIC / "source",
        "checkpoint/experiment_adapters": PUBLIC / "qwen_adapter_fair_comparison/training",
        "thesis_materials/project_guide": PROJECT / "doc/project_guide",
        "thesis_materials/originals/初稿.docx": PROJECT / "初稿.docx",
        "thesis_materials/originals/论文1-3章初稿.md": PROJECT / "论文1-3章初稿.md",
        "thesis_materials/existing_draft": OLD / "thesis_public_datasets_draft_v1",
        "thesis_materials/tables": PUBLIC / "tables",
        "thesis_materials/figures": PUBLIC / "figures",
        "thesis_materials/historical_handoffs": OLD / "thesis",
    }


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def snapshot(roots):
    """不沿软链接遍历；原始数据和共享卷只建立入口，不复制或移动。"""
    rows = []
    for root in roots:
        for parent, directories, names in os.walk(root, followlinks=False):
            for name in sorted(directories + names):
                path = Path(parent) / name
                if path.is_symlink():
                    rows.append(dict(path=str(path), type="symlink", target=os.readlink(path),
                                     target_existed=path.exists()))
                elif path.is_file():
                    rows.append(dict(path=str(path), type="file", size=path.stat().st_size,
                                     sha256=digest(path)))
    return rows


def verify(rows, migration_rows, entry_links):
    errors = []
    existing_broken_links = 0
    for row in rows:
        path = Path(row["path"])
        if row["type"] == "file":
            if not path.is_file() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
                errors.append(dict(path=str(path), error="file_changed_or_missing"))
        else:
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                errors.append(dict(path=str(path), error="original_symlink_changed"))
            elif row["target_existed"] and not path.exists():
                errors.append(dict(path=str(path), error="newly_broken_original_symlink"))
            elif not row["target_existed"]:
                existing_broken_links += 1
    for old, new in migration_rows:
        if not old.is_symlink() or not new.is_dir() or old.resolve() != new.resolve():
            errors.append(dict(path=str(old), error="migration_alias_mismatch"))
    for relative, target in entry_links.items():
        path = PROJECT / relative
        if not path.is_symlink() or not path.exists() or path.resolve() != target.resolve():
            errors.append(dict(path=str(path), error="entry_link_mismatch"))
    return dict(passed=not errors, original_regular_files=sum(r["type"] == "file" for r in rows),
                original_regular_bytes=sum(r.get("size", 0) for r in rows),
                original_symlinks=sum(r["type"] == "symlink" for r in rows),
                previously_broken_links=existing_broken_links, migrations=len(migration_rows),
                new_entry_links=len(entry_links), errors=errors,
                experiment_rerun=False, original_labels_or_metrics_changed=False)


def separate_generated_results():
    """将旧程序写在输入树里的exp_*预测结果迁回outputs，原位置留入口。"""
    if not (AUDIT / "verification.json").exists():
        raise RuntimeError("先执行 --apply 完成输入目录整理")
    audit = AUDIT / "prediction_separation"
    if audit.exists():
        raise FileExistsError("已有预测目录分离记录，请使用 --verify")
    migration_rows = []
    for source, group in [
        (PROJECT / "dataset/processed/nuscenes/validation", "nuscenes_validation"),
        (PROJECT / "dataset/processed/preparation_history", "preparation_history"),
    ]:
        for parent, directories, _ in os.walk(source, followlinks=False):
            for name in list(directories):
                if name.startswith("exp_"):
                    path = Path(parent) / name
                    directories.remove(name)
                    if not path.is_symlink():
                        target = OLD / "reorganized_results_20261118" / group / path.relative_to(source)
                        if target.exists():
                            raise FileExistsError(target)
                        migration_rows.append((path, target))
    entry_links = {"outputs/results/nuscenes/validation/detections":
                   OLD / "reorganized_results_20261118/nuscenes_validation/converted"}
    for relative in entry_links:
        path = PROJECT / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    audit.mkdir()
    write_json(audit / "plan.json", dict(
        moves=[dict(old=str(a), new=str(b)) for a, b in migration_rows],
        links={a: str(b) for a, b in entry_links.items()}))
    rows = snapshot([a for a, _ in migration_rows])
    write_json(audit / "before_files.json", rows)
    for old, new in migration_rows:
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.stat().st_dev != new.parent.stat().st_dev:
            raise ValueError("只允许同盘重命名")
        old.rename(new)
        try:
            old.symlink_to(new, target_is_directory=True)
        except OSError:
            new.rename(old)
            raise
    for relative, target in entry_links.items():
        (PROJECT / relative).symlink_to(target, target_is_directory=True)
    report = verify(rows, migration_rows, entry_links)
    write_json(audit / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--separate-generated-results", action="store_true")
    args = parser.parse_args()
    if args.separate_generated_results:
        separate_generated_results()
    migration_rows = moves()
    entry_links = links()
    plan = dict(moves=[dict(old=str(a), new=str(b)) for a, b in migration_rows],
                links={str(PROJECT / a): str(b) for a, b in entry_links.items()},
                strategy="same_filesystem_rename_then_old_path_symlink; no deletion; original bytes unchanged")
    if args.verify:
        rows = json.loads((AUDIT / "before_files.json").read_text())
        report = verify(rows, migration_rows, entry_links)
        supplementary = AUDIT / "prediction_separation"
        if (supplementary / "plan.json").exists():
            extra_plan = json.loads((supplementary / "plan.json").read_text())
            extra_rows = json.loads((supplementary / "before_files.json").read_text())
            report["prediction_separation"] = verify(
                extra_rows, [(Path(a["old"]), Path(a["new"])) for a in extra_plan["moves"]],
                {a: Path(b) for a, b in extra_plan["links"].items()})
            report["passed"] = report["passed"] and report["prediction_separation"]["passed"]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["passed"] else 1)
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if AUDIT.exists():
        raise FileExistsError(f"已有迁移记录，不重复执行；用 --verify：{AUDIT}")
    for old, new in migration_rows:
        if new.exists() or new.is_symlink():
            raise FileExistsError(new)
        # 头一步迁移后才出现的源目录在执行前不要求存在。
        if old.is_relative_to(PROJECT / "dataset/processed/preparation_history"):
            continue
        if not old.is_dir() or old.is_symlink():
            raise ValueError(f"源路径不是预期的真实目录：{old}")
    for relative, target in entry_links.items():
        path = PROJECT / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if not target.exists() and not str(target).startswith(str(PROJECT / "dataset/processed")) and target != AUDIT and target != PROJECT / "doc/project_guide":
            raise FileNotFoundError(target)
    AUDIT.mkdir(parents=True)
    write_json(AUDIT / "plan.json", plan)
    (AUDIT / "git_status_before.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT))
    (AUDIT / "git_diff_before.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=PROJECT))
    initial_roots = [a for a, _ in migration_rows if not a.is_relative_to(PROJECT / "dataset/processed/preparation_history")]
    rows = snapshot(initial_roots)
    write_json(AUDIT / "before_files.json", rows)
    print(f"已记录 {len(rows)} 项，开始同盘迁移。", flush=True)
    for old, new in migration_rows:
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.stat().st_dev != new.parent.stat().st_dev:
            raise ValueError("只允许同一文件系统内重命名")
        old.rename(new)
        try:
            old.symlink_to(new, target_is_directory=True)
        except OSError:
            new.rename(old)
            raise
    for relative, target in entry_links.items():
        path = PROJECT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=target.is_dir())
    report = verify(rows, migration_rows, entry_links)
    write_json(AUDIT / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
