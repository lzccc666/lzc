"""候选包补充白名单/未分配状态/同意图改写检查，不制造人审结果。"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from jsonschema import Draft202012Validator

from conceptgraph.project_paths import PROJECT_ROOT


def main():
    folder = PROJECT_ROOT / "dataset/qwen/multiview_task_20261123_intent_candidates_v1"
    report = PROJECT_ROOT / "outputs/reports/multiview_task_iteration_20261123/qwen"
    fields = {"sample_id", "paraphrase_family_id", "instruction", "intent_draft", "ambiguity_kind", "package_id", "source", "draft_status", "human_review_status", "human_review", "scene", "log_token", "episode_id", "issue_frame_idx", "issue_time_s", "split", "grounding_status", "training_eligible", "instruction_sha256"}
    rows = [json.loads(s) for s in (folder / "candidates.jsonl").read_text().splitlines()]
    schema = json.loads((folder / "language_intent.schema.json").read_text())
    validator = Draft202012Validator(schema)
    signatures = defaultdict(set)
    for r in rows:
        assert set(r) == fields
        validator.validate(r["intent_draft"])
        assert all(r[k] is None for k in ("scene", "log_token", "episode_id", "issue_frame_idx", "issue_time_s"))
        assert r["split"] == "unassigned" and r["human_review_status"] == "pending" and r["training_eligible"] is False
        assert r["source"]["kind"] == "programmatic_template_candidate" and r["source"]["human_authored"] is False
        assert hashlib.sha256(r["instruction"].encode()).hexdigest() == r["instruction_sha256"]
        signatures[json.dumps(r["intent_draft"], sort_keys=True, ensure_ascii=False)].add(r["paraphrase_family_id"])
    same_intent = [sorted(v) for v in signatures.values() if len(v) > 1]
    with (folder / "candidates.csv").open(encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 72 and [r["instruction"] for r in rows] == [r["instruction"] for r in csv_rows]
    manifest = json.loads((folder / "manifest.json").read_text())
    source_schema = Path(manifest["schema_source"])
    assert source_schema.read_bytes() == (folder / "language_intent.schema.json").read_bytes()
    with zipfile.ZipFile(report / "Qwen任务意图_72条待审核离线包.zip") as z:
        assert z.testzip() is None
        for path in folder.iterdir():
            if path.is_file():
                assert z.read(path.name) == path.read_bytes()
    result = dict(passed=True, rows=72, explicit_candidate_field_whitelist=True,
        schema_valid_rows=72, CSV_matches_JSONL=True, source_schema_unchanged=True,
        scene_log_episode_time_unassigned_rows=72, pending_rows=72, training_eligible_rows=0,
        same_intent_template_family_groups=same_intent,
        split_requirement="Before any split, merge these same-intent template families and any human-identified paraphrase families into one linked component; also group scene/log/episode. No split has been assigned.",
        archive_files_byte_equal=True, human_annotations_created=0, model_calls=0, GPU_calls=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (report / "candidate_package_final_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
