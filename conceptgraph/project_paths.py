"""整理后的新实验路径；历史复现模块的路径保持不变。"""
from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "outputs/results"
LEGACY_REMOTE_ROOT = Path("/root/autodl-tmp/lizichong")


def resolve_project_path(value: str | Path) -> Path:
    """Resolve frozen remote project references against this checkout."""
    path = Path(value)
    if path.is_relative_to(LEGACY_REMOTE_ROOT):
        return PROJECT_ROOT / path.relative_to(LEGACY_REMOTE_ROOT)
    return path


def experiment_directory(dataset: str, split: str, run_id: str, *, create=False) -> Path:
    """新实验一律分类落盘；拒绝路径穿越与覆盖已存在的run。"""
    for value in (dataset, split, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError(f"非法实验路径字段：{value!r}")
    path = RESULTS_ROOT / dataset / split / "iterations" / run_id
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path
