"""Import versioned language-review evidence without granting training or task success."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


EXPORT_KEYS = {
    "package_id", "manifest_sha256", "intent_schema_sha256", "exported_at",
    "identity_verified", "training_export_enabled", "trajectory_evaluation_status", "review_events",
}
EVENT_KEYS = {
    "sample_id", "source_instruction_sha256", "reviewed_instruction", "intent_draft",
    "instruction_changed", "requires_intent_recheck", "decision", "reviewed_at", "notes", "origin",
}
DECISIONS = {"pending", "approve", "needs_revision", "reject"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON重复字段：{key}")
        result[key] = value
    return result


def load_json(data: bytes):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"非法JSON数值：{value}")))


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label}字段不匹配")


def _timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}不是时间字符串")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}时间格式错误") from exc
    if stamp.tzinfo is None:
        raise ValueError(f"{label}缺时区")
    return stamp


def load_source_package(package: Path) -> dict:
    manifest_path = package / "manifest.json"
    manifest = load_json(manifest_path.read_bytes())
    schema_path = package / "simple_intent.schema.json"
    candidates_path = package / "candidates.jsonl"
    for path, key in [(schema_path, "intent_schema_sha256"), (candidates_path, "candidates_sha256"),
                      (package / "source_history.jsonl", "source_history_sha256")]:
        if file_sha256(path) != manifest[key]:
            raise ValueError(f"源包指纹不匹配：{path.name}")
    schema = load_json(schema_path.read_bytes())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    rows = [load_json(line.encode("utf-8")) for line in candidates_path.read_text().splitlines() if line]
    by_id = {}
    for row in rows:
        if row["sample_id"] in by_id:
            raise ValueError("源候选ID重复")
        if row["package_id"] != manifest["package_id"]:
            raise ValueError("源候选包ID错误")
        if sha256_bytes(row["instruction"].encode("utf-8")) != row["instruction_sha256"]:
            raise ValueError("源候选指令指纹错误")
        validator.validate(row["intent_draft"])
        if row["human_review_status"] != "pending" or row["training_eligible"] is not False:
            raise ValueError("源包不是原始待审包")
        by_id[row["sample_id"]] = row
    if len(rows) != manifest["rows"]:
        raise ValueError("源包候选数不匹配")
    return dict(path=package, manifest=manifest, manifest_sha256=file_sha256(manifest_path),
                schema_sha256=file_sha256(schema_path), validator=validator, rows=rows, by_id=by_id)


def validate_export(payload: dict, source: dict) -> dict:
    _exact(payload, EXPORT_KEYS, "审核导出")
    manifest = source["manifest"]
    for actual, expected, label in [
        (payload["package_id"], manifest["package_id"], "包ID"),
        (payload["manifest_sha256"], source["manifest_sha256"], "manifest"),
        (payload["intent_schema_sha256"], source["schema_sha256"], "schema"),
    ]:
        if actual != expected:
            raise ValueError(f"审核{label}不匹配")
    if payload["identity_verified"] is not False or payload["training_export_enabled"] is not False:
        raise ValueError("审核不能授予身份或训练权限")
    if payload["trajectory_evaluation_status"] != "not_evaluated":
        raise ValueError("语言审核不能授予轨迹成绩")
    exported_at = _timestamp(payload["exported_at"], "导出")
    if not isinstance(payload["review_events"], list):
        raise ValueError("审核事件不是列表")
    events = {}
    for event in payload["review_events"]:
        _exact(event, EVENT_KEYS, "审核事件")
        sample_id = event["sample_id"]
        if not isinstance(sample_id, str) or sample_id not in source["by_id"] or sample_id in events:
            raise ValueError("审核ID未知或重复")
        original = source["by_id"][sample_id]
        if event["source_instruction_sha256"] != original["instruction_sha256"]:
            raise ValueError("审核指令来源指纹错误")
        instruction = event["reviewed_instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("审核指令为空")
        changed = instruction != original["instruction"]
        if event["instruction_changed"] is not changed or event["requires_intent_recheck"] is not changed:
            raise ValueError("指令改写标记错误")
        if event["decision"] not in DECISIONS or not isinstance(event["notes"], str):
            raise ValueError("审核决定或备注无效")
        if event["origin"] != "browser_explicit_save":
            raise ValueError("审核来源不匹配")
        if _timestamp(event["reviewed_at"], "审核") > exported_at:
            raise ValueError("审核晚于导出")
        source["validator"].validate(event["intent_draft"])
        if event["intent_draft"] != original["intent_draft"]:
            raise ValueError("导入不能静默改写原意图草案")
        events[sample_id] = copy.deepcopy(event)
    return events


def form_split_groups(rows: list) -> tuple[dict, list]:
    parents = {row["sample_id"]: row["sample_id"] for row in rows}

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    seen = {}
    for row in rows:
        keys = [("family", row["paraphrase_family_id"]),
                ("intent", json.dumps(row["intent_draft"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))]
        keys.extend((field, row[field]) for field in ("scene", "log_token", "episode_id") if row.get(field) is not None)
        for key in keys:
            if key in seen:
                union(row["sample_id"], seen[key])
            else:
                seen[key] = row["sample_id"]
    components = defaultdict(list)
    for row in rows:
        components[find(row["sample_id"])].append(row)
    by_id, groups = {}, []
    for component in sorted(components.values(), key=lambda entries: min(r["sample_id"] for r in entries)):
        families = sorted({r["paraphrase_family_id"] for r in component})
        group_id = "language_group__" + "__".join(families)
        members = sorted(r["sample_id"] for r in component)
        for sample_id in members:
            by_id[sample_id] = group_id
        groups.append(dict(split_group_id=group_id, families=families, sample_ids=members, split="unassigned"))
    return by_id, groups


def derive_records(payload: dict, source: dict, raw_sha256: str) -> dict:
    events = validate_export(payload, source)
    group_by_id, groups = form_split_groups(source["rows"])
    ledger, approved = [], []
    task_counts = defaultdict(Counter)
    counts = Counter()
    for original in source["rows"]:
        sample_id = original["sample_id"]
        event = events.get(sample_id)
        decision = event["decision"] if event else "unreviewed"
        counts[decision] += 1
        task_counts[original["intent_draft"]["action"]][decision] += 1
        status = {"approve": "language_intent_approved", "reject": "not_adopted",
                  "needs_revision": "pending_revision", "pending": "pending_review",
                  "unreviewed": "unreviewed"}[decision]
        if event and event["requires_intent_recheck"]:
            status = "pending_intent_recheck"
        row = dict(
            sample_id=sample_id, source_package_id=payload["package_id"],
            paraphrase_family_id=original["paraphrase_family_id"], split_group_id=group_by_id[sample_id],
            instruction=original["instruction"], instruction_sha256=original["instruction_sha256"],
            intent=copy.deepcopy(original["intent_draft"]), source_kind=original["source_kind"],
            review_decision=decision, language_review_status=status, review_event=event,
            review_source=dict(kind="user_submitted_chat_attachment", sha256=raw_sha256,
                               source_manifest_sha256=payload["manifest_sha256"],
                               source_schema_sha256=payload["intent_schema_sha256"]),
            identity_verified=payload["identity_verified"], training_export_enabled=False,
            training_eligible=False, split="unassigned", scene=None, log_token=None, episode_id=None,
            grounding_status="unbound_unknown", trajectory_evaluation_status="not_evaluated",
            natural_user_original_instruction=False, review_applies_to="this_sample_only",
        )
        ledger.append(row)
        if status == "language_intent_approved":
            approved.append(copy.deepcopy(row))
    return dict(ledger=ledger, approved=approved, groups=groups, counts=dict(counts),
                task_counts={task: dict(count) for task, count in task_counts.items()})


def rejection_checks(payload: dict, source: dict) -> dict:
    def invalid(mutator):
        bad = copy.deepcopy(payload)
        mutator(bad)
        try:
            validate_export(bad, source)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            return {"rejected": True, "error_type": type(exc).__name__}
        raise AssertionError("非法审核记录未被拒绝")

    return {
        "wrong_manifest": invalid(lambda bad: bad.update(manifest_sha256="0" * 64)),
        "wrong_schema": invalid(lambda bad: bad.update(intent_schema_sha256="0" * 64)),
        "duplicate_event": invalid(lambda bad: bad["review_events"].append(copy.deepcopy(bad["review_events"][0]))),
        "wrong_instruction_hash": invalid(lambda bad: bad["review_events"][0].update(source_instruction_sha256="0" * 64)),
        "wrong_change_flag": invalid(lambda bad: bad["review_events"][0].update(instruction_changed=True)),
        "training_permission": invalid(lambda bad: bad.update(training_export_enabled=True)),
        "trajectory_success": invalid(lambda bad: bad.update(trajectory_evaluation_status="passed")),
        "extra_completion_field": invalid(lambda bad: bad["review_events"][0]["intent_draft"].update(termination_request={})),
    }
