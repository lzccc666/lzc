"""Validate prepared episode evidence; never infer missing human labels or score tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def validate(root):
    protocol = read(root / "episode_protocol.json")
    task = read(root / "task_pending_review.json")
    frames = lines(root / "inference_frames.jsonl")
    labels = lines(root / "evaluation_labels_pending.jsonl")
    errors = []
    missing = []
    if len(frames) != protocol["frames"] or len(labels) != len(frames):
        errors.append("帧或评估标签行数不符")
    if [r["frame_idx"] for r in frames] != list(range(protocol["frames"])):
        errors.append("帧索引不连续")
    times = [r["camera_timestamp_s"] for r in frames]
    if any(a >= b for a, b in zip(times, times[1:])):
        errors.append("时间未严格递增")
    for frame in frames:
        image = Path(frame["rgb_path"])
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != frame["rgb_sha256"]:
            errors.append(f"图像缺失或改变f{frame['frame_idx']}")
        for node in frame["predicted_nodes"]:
            if (node.get("last_seen_frame", frame["frame_idx"]) > frame["frame_idx"] or
                    node.get("timestamp_s", 0) > frame["camera_timestamp_s"] + 1e-6):
                errors.append("预测图含未来观测")
            if any(k in node for k in ("instance_token", "truth_index", "acceptable_instances", "selected_correct")):
                errors.append("预测节点中出现评价专用字段")
    if not task.get("human_reviewed") or not task.get("reviewer") or not task.get("review_timestamp"):
        missing.append("真人任务审核记录")
    if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        missing.append("一条整段自然指令")
    if not isinstance(task.get("issue_frame_idx"), int) or not 0 <= task["issue_frame_idx"] < len(frames):
        missing.append("指令发出帧")
    if not task.get("human_target_identity") or not task.get("target_identity_audit_complete"):
        missing.append("目标身份及跨帧身份审核")
    if task.get("uses_color") is None or (task.get("uses_color") and not task.get("color_review_complete")):
        missing.append("颜色是否使用及必要时的颜色复核")
    if any(not r.get("human_reviewed") or not r.get("reviewer") or not r.get("review_timestamp")
           or not isinstance(r.get("target_visible"), bool) or not isinstance(r.get("target_absent"), bool)
           or not isinstance(r.get("target_ambiguous"), bool) for r in labels):
        missing.append("逐时刻可见、缺席和歧义审核")
    if protocol["metric_protocol"].get("unfrozen_parameters"):
        missing.append("运行前冻结锁定状态机、可定位与路线判据")
    return {"status": "invalid_inputs" if errors else "blocked_human_annotation" if missing else "inputs_ready_not_executed",
            "input_integrity_passed": not errors, "frames_checked": len(frames),
            "errors": errors, "missing": missing, "model_calls": 0,
            "selector_calls": 0, "planner_calls": 0, "episode_metrics_produced": False,
            "note": "仅验证输入完整性；ready不等于已执行或已取得成绩"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.input_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(text)
    print(text, end="")
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
