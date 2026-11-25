"""CPU-only independent input freeze for the bounded 2026-11-24 iteration."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def main():
    out = ROOT / "outputs/reports/iteration_independent_audit_20261124"
    out.mkdir(parents=True, exist_ok=True)
    run = experiment_directory("qwen", "validation", "iteration_independent_audit_20261124_inputs_v1", create=True)
    registration_path = ROOT / "outputs/reports/mapping_iteration_20261124/mapnext_20261124_registration.json"
    reg = json.loads(registration_path.read_text())
    protected = dict(reg["protected_sha256"])
    paths = [
        ROOT / "conceptgraph/hydra_configs/semantic_navigation.yaml",
        ROOT / "conceptgraph/vlm/qwen_semantic_delegate.py",
        ROOT / "conceptgraph/navigation/pipeline.py",
        ROOT / "dataset/qwen/real_graph_v003/manifest.json",
        ROOT / "dataset/qwen/real_graph_v003/train.jsonl",
        ROOT / "dataset/qwen/real_graph_v003/validation.jsonl",
        ROOT / "dataset/qwen/paper_qwen_20261123_pilot_v1/manifest.json",
        ROOT / "dataset/qwen/paper_qwen_20261123_pilot_v1/train.jsonl",
        ROOT / "dataset/qwen/paper_qwen_20261123_pilot_v1/validation.jsonl",
        ROOT / "dataset/qwen/command_review_import_20261124_v2/manifest.json",
        ROOT / "dataset/qwen/command_review_import_20261124_v2/approved_language_intents.jsonl",
        ROOT / "checkpoint/Qwen2-VL-2B-Instruct/config.json",
        ROOT / "checkpoint/qwen_lora_semantic_delegate/adapter_config.json",
    ]
    protected.update({str(p): sha(p) for p in paths})
    assert all(sha(p) == s for p, s in protected.items())
    source = Path(__file__).resolve()
    snap = run / "source_snapshot" / "scripts" / source.name
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snap)
    pilot = {}
    sets = {}
    for split in ("train", "validation"):
        rs = rows(ROOT / "dataset/qwen/paper_qwen_20261123_pilot_v1" / (split + ".jsonl"))
        sets[split] = {
            "log": {r["log_token"] for r in rs},
            "scene": {r["scene_id"] for r in rs},
            "frame": {(r["scene_id"], r["frame_idx"]) for r in rs},
            "sample": {r["sample_id"] for r in rs},
            "query": {r["query_text"] for r in rs},
        }
        pilot[split] = dict(
            requests=len(rs),
            response_selection=sum(r["response"]["selected_node_id"] is not None for r in rs),
            response_null=sum(r["response"]["selected_node_id"] is None for r in rs),
            ordinary_missing=[r["sample_id"] for r in rs if r["official_positive"] and not r["controlled_fault"] and not r["candidate_target_coverage"]],
            strata=dict(Counter("controlled_fault" if r["controlled_fault"] else "ordinary_covered" if r["official_positive"] and r["candidate_target_coverage"] else "ordinary_missing" if r["official_positive"] else "true_no_match" for r in rs)),
        )
    overlaps = {k: len(sets["train"][k] & sets["validation"][k]) for k in sets["train"]}
    assert all(overlaps[k] == 0 for k in ("log", "scene", "frame", "sample"))
    result = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        audit_deadline_utc="2026-11-24T08:08:00+00:00", maximum_new_bytes=64 * 2**20,
        protected_sha256=protected, protected_count=len(protected), protected_unchanged=True,
        mapping_registration=str(registration_path), mapping_registration_sha256=sha(registration_path),
        mapping_protocol=reg["evaluation"], mapping_validation_and_regression_previously_seen=True,
        pilot=pilot, pilot_overlap=overlaps,
        pilot_overlap_scope="shared generic query templates across independent physical logs; not language-unseen generalization",
        manifest_expected_null_issue="both splits have 20 null responses: 12 controlled fault + 4 true no-match + 4 missing positive; old expected_null=16 omits the latter 4 in its auxiliary count",
        official_GT_in_inference_allowed=False, language16_scope="already-seen language regression only, unbound to scene/trajectory",
        model_calls=0, GPU_calls=0, training_calls=0, source_sha256=sha(source), source_snapshot=str(snap),
    )
    (out / "initial_input_freeze.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (run / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("protected_count", "protected_unchanged", "pilot_overlap", "pilot")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
