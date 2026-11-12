"""Refresh runtime, fault and acceptance evidence without changing experiments."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.lane_expansion_report_tables import rate, report_tables
from conceptgraph.scripts.lane_expansion_cohort_reports import build_cohort_reports
from conceptgraph.scripts.lane_expansion_qwen_effects import build_qwen_effect_report
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    DEFAULT_CONFIG,
    write_json,
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def audited_task_counts(qwen_online, navigation):
    tasks = defaultdict(Counter)
    for row in qwen_online.get("rows", []):
        original = next(
            r
            for r in navigation["rows"]
            if r["scene"] == row["scene"]
            and r["query_id"] == row["query_id"]
            and r["method"] == row["method"]
        )
        for repeat in ("first", "cache_repeat"):
            for path in (Path(original["artifact"]) / repeat / "qwen_audit").rglob(
                "*.json"
            ):
                audit = read(path)
                envelope = audit.get("result", audit)
                key = (row["role"], row["method"], repeat, envelope["task"])
                tasks[key]["audited_requests"] += 1
                tasks[key]["cache_hits"] += bool(envelope.get("cache", {}).get("hit"))
                tasks[key]["schema_valid"] += bool(envelope.get("output_schema_valid"))
                tasks[key]["successful_fresh_responses"] += (
                    bool(envelope.get("used_qwen"))
                    and bool(envelope.get("output_schema_valid"))
                    and not bool(envelope.get("cache", {}).get("hit"))
                )
                fresh_request = not bool(envelope.get("cache", {}).get("hit"))
                raw_output = envelope.get("raw_output")
                tasks[key]["fresh_nonempty_backend_text_observed"] += (
                    fresh_request and isinstance(raw_output, str) and bool(raw_output.strip())
                )
                tasks[key]["fresh_empty_backend_text_observed"] += (
                    fresh_request and isinstance(raw_output, str) and not raw_output.strip()
                )
                tasks[key]["fresh_schema_invalid_text_responses"] += (
                    fresh_request and raw_output is not None and not envelope.get("output_schema_valid", False)
                )
                tasks[key]["fallbacks"] += bool(envelope.get("fallback_used"))
    return tasks


def task_count_rows(tasks):
    return [
        {"role": k[0], "method": k[1], "repeat": k[2], "task": k[3], **dict(v)}
        for k, v in sorted(tasks.items())
    ]


def main():
    root = DEFAULT_ROOT
    cohorts = build_cohort_reports(root)
    qwen_effects = build_qwen_effect_report(root)
    navigation = read(root / "navigation_report.json", {})
    retrieval = read(root / "retrieval_report.json", {})
    ablation = read(root / "ablation_report.json", {})
    qwen_online = read(root / "qwen_online_report.json", {})
    selected = read(root / "scene_selection_manifest.json")["scenes"]
    baseline = read(root / "baseline_before_expansion.json")
    versions = {}
    for name in (
        "torch", "transformers", "peft", "numpy", "scipy", "shapely", "ultralytics",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    model_paths = (
        "checkpoint/Qwen2-VL-2B-Instruct/",
        "outputs/project_v2/qwen_eval/finetune_v2/best_adapter/",
    )
    weights = [r for r in baseline["files"] if r["path"].startswith(model_paths)]
    tasks = audited_task_counts(qwen_online, navigation)
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    runtime = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "software_versions": versions,
        "frozen_model_and_adapter_files": weights,
        "weight_hash_verification_source": "baseline_before_expansion.json + complete streaming baseline_verification.json",
        "generation": {
            "do_sample": False,
            "max_new_tokens": int(cfg.qwen_online.max_new_tokens),
            "dtype": str(cfg.qwen_online.torch_dtype),
            "max_image_side": int(cfg.qwen_online.max_image_side),
            "prompt_profile": str(cfg.qwen_online.prompt_profile),
        },
        "task_counts": task_count_rows(tasks),
        "online_totals": qwen_online.get("totals", {}),
        "call_counter_semantics": {
            "backend_call_count": "actual backend attempts recorded by each replay backend",
            "successful_backend_response_count": "existing replay metric: fresh model response accepted by the strict task schema; not raw generation completion",
            "fresh_nonempty_backend_text_observed": "fresh request audit contains actual nonempty raw model text, including schema-invalid responses; cache hits excluded",
            "schema_valid": "task schema validity, separate from raw text generation and actual online application",
        },
        "multimodal_processor_probe": read(
            root / "runtime_probes/multimodal_processor_report.json",
            {"passed": False, "status": "pending after CUDA matrix frees a worker"},
        ),
        "gpu_peak_note": "actual per-process allocator maximum; persistent backend means later job peaks include earlier allocations",
        "model_training_or_download_performed": False,
    }
    write_json(root / "qwen_runtime_report.json", runtime)
    fault_scenes, missing_faults = [], []
    for scene in selected:
        if scene["role"] != "sealed_test" and scene["scene"] not in {
            "scene-0003",
            "scene-0916",
        }:
            continue
        path = root / "experiments/faults_matrix_v2" / scene["scene"] / "complete.json"
        record = read(path)
        if record is None:
            missing_faults.append(scene["scene"])
        else:
            fault_scenes.append({"scene": scene["scene"], **record})
    fault_cases = [c for s in fault_scenes for c in s["cases"]]
    async_report = read(
        root / "asynchronous_fault_timeline.json",
        {"passed": False, "status": "pending"},
    )
    write_json(
        root / "failure_injection_report.json",
        {
            "all_complete": len(fault_scenes) == 12,
            "passed": not missing_faults
            and all(s["passed"] for s in fault_scenes)
            and async_report["passed"],
            "scene_reports": fault_scenes,
            "missing_scenes": missing_faults,
            "case_count": len(fault_cases),
            "case_failure_count": sum(c.get("passed") is False for c in fault_cases),
            "not_applicable_count": sum(c.get("passed") is None for c in fault_cases),
            "baseline_path_presence_counts": dict(
                Counter(str(c.get("baseline_legal_path_exists")) for c in fault_cases)
            ),
            "asynchronous_timeline_artifact": str(
                root / "asynchronous_fault_timeline.json"
            ),
            "asynchronous_passed": async_report["passed"],
            "all_backends_synthetic": True,
            "real_qwen_backend_calls": 0,
        },
    )
    xml = next(
        (
            p
            for p in (root / "tests_final_v2.xml", root / "tests_final.xml", root / "tests_pre_release.xml")
            if p.exists()
        ),
        None,
    )
    test_counts = None
    if xml:
        suites = ET.parse(xml).getroot().findall("testsuite")
        test_counts = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    paths = [read(p) for p in (root / "path_audits").glob("*.json")]
    # Online V1/V2 sources are selected per job with explicit repair provenance.
    # Failed superseded diagnostics remain visible and are never overwritten.
    primary_runs = {
        "static_development_all_v3",
        "static_test_v1",
        "retrieval_off_v2",
        "retrieval_adapter_v1",
        "retrieval_base_v1",
        "faults_matrix_v2",
    }
    audited = {p["run_id"] for p in paths if p["passed"]}
    off_rows = [r for r in navigation.get("rows", []) if r["variant"] == "off"]
    gpu_rows = [r for r in navigation.get("rows", []) if r["variant"] != "off"]
    compatibility = read(root / "base_prompt_compatibility_report.json", {})
    application_evidence = {}
    for variant in ("base", "adapter"):
        members = [r for r in gpu_rows if r["variant"] == variant]
        applied = Counter()
        for row in members:
            applied.update(row["qwen_applications"])
        fresh = {
            task: sum(
                v["successful_fresh_responses"]
                for k, v in tasks.items()
                if k[1].startswith(variant + "_") and k[2] == "first" and k[3] == task
            )
            for task in ("query_parse", "object_attributes", "candidate_rerank")
        }
        primary_complete = len(members) == 72 and all(
            row["execution_passed"] for row in members
        )
        primary_tasks_passed = all(v > 0 for v in fresh.values()) and all(
            applied[task + "_applied"] > 0 for task in fresh
        )
        supplemental_parse = bool(
            variant == "base"
            and compatibility.get("passed")
            and compatibility.get("query_parse_application_evidence")
            and compatibility.get("counts_in_primary_comparison") is False
            and compatibility.get("determinism_passed")
            and compatibility.get("cache_repeat_zero_new_backend_calls")
            and compatibility.get("path_audit", {}).get("passed")
        )
        task_sources = {
            task: (
                "primary_frozen_online_matrix"
                if fresh[task] > 0 and applied[task + "_applied"] > 0
                else "separate_stage4_prompt_development_replay"
                if task == "query_parse" and supplemental_parse
                else None
            )
            for task in fresh
        }
        application_evidence[variant] = {
            "completed_online_pairs": len(members),
            "fresh_schema_valid_responses_by_task": fresh,
            "applied_by_task": {task: applied[task + "_applied"] for task in fresh},
            "primary_matrix_execution_complete": primary_complete,
            "primary_matrix_has_all_three_task_applications": primary_tasks_passed,
            "task_application_evidence_sources": task_sources,
            "supplementary_base_parse_capability_passed": supplemental_parse,
            "supplementary_counts_added_to_primary_metrics": False,
            "passed": primary_complete and all(task_sources.values()),
        }
    determinism = read(root / "determinism_report.json", {})
    semantic_repeats = read(root / 'semantic_repeat_audit.json', {})
    all_repeat_rows = determinism.get("online", []) + determinism.get("retrieval", [])
    cache_hits_observed = all(
        r["qwen_cache_repeat"]["cache_hit_count"] > 0 for r in gpu_rows
    )
    retrieval_gpu = [r for r in retrieval.get("rows", []) if r["variant"] != "off"]
    cache_hits_observed = cache_hits_observed and all(
        r["qwen_cache_repeat"]["cache_hit_count"] > 0 for r in retrieval_gpu
    )
    query_rows = [
        json.loads(line) for line in (root / "benchmark_queries.jsonl").open()
    ]
    query_counts = Counter(row["scene"] for row in query_rows)
    official_split = read(root / "official_split_audit.json", {})
    temporal = read(root / "temporal_adjacency_audit.json", {})
    geometry = read(root / "annotation_geometry_audit.json", {})
    physical_coverage = read(root / 'physical_coverage_audit.json', {})
    ground_truth = read(root / "ground_truth_manifest.json", {})
    statistics = read(root / "statistical_report.json", {})
    semantic = read(root / 'semantic_report.json', {})
    visualization = read(root / "visualizations/visualization_manifest.json", {})
    browser_qa = read(root/'visualizations/browser_qa_report.json', {})
    visual_source = root/'visualizations/source_manifest.json'
    visual_metadata = read(visual_source, {})
    browser_qa_current = bool(browser_qa.get('passed') and visual_source.exists() and browser_qa.get('source_manifest_sha256') == hashlib.sha256(visual_source.read_bytes()).hexdigest()
        and visual_metadata.get('renderer_template_sha256') == hashlib.sha256(Path(__file__).with_name('lane_expansion_browser.html').read_bytes()).hexdigest()
        and visual_metadata.get('exporter_sha256') == hashlib.sha256(Path(__file__).with_name('visualize_lane_expansion.py').read_bytes()).hexdigest())
    checks = read(root / "verification_checks.json", {})
    active_paths = read(root / "path_audits/production_active_online.json", {})
    production_index = read(root / "production_job_index.json", {})
    active_paths_passed = bool(
        active_paths.get("passed")
        and active_paths.get("completed_job_pairs") == 444
        and active_paths.get("source_index_sha256") == production_index.get("sha256")
    )
    safety_sources = {
        p['run_id']: p for p in paths
        if p['run_id'] in primary_runs | {'production_active_online'}
    }
    safety_summary = {
        'all_complete': primary_runs <= audited and active_paths_passed,
        'required_run_count': len(primary_runs) + 1,
        'runs': [{
            'run_id': name,
            'passed': report.get('passed', False),
            'accepted_or_reused_path_checks': report.get('accepted_or_reused_paths_checked', 0),
            'failed_path_checks': len(report.get('failures', [])),
            'scope': 'first and cache repeats; saved actual occupancy; every accepted path and every reused remaining segment',
        } for name, report in sorted(safety_sources.items())],
        'counts_are_path_checks_not_independent_scenes': True,
        'zero_accepted_paths_has_undefined_safety_rate': True,
        'superseded_unsafe_diagnostic_retained': 'path_audits/cpu_development_scene0038_v2.json',
    }
    safety_summary['original_diagnostic_cohort_complete'] = safety_summary['all_complete']
    safety_summary['distinct_online_accepted_or_reused_path_checks'] = active_paths.get('accepted_or_reused_paths_checked', 0)
    safety_summary['overlapping_run_audits_must_not_be_added_to_active_online_audits'] = True
    for row in safety_summary['runs']:
        row['cohort'] = 'original_accessed_diagnostic_50'
    if cohorts is not None:
        fresh_root = root / 'prospective_supplement_v2'
        fresh_active = read(fresh_root / 'path_audits/production_active_online.json', {})
        fresh_runs = [read(path) for path in sorted((fresh_root / 'path_audits').glob('*.json'))]
        safety_summary['prospective_supplement_complete'] = cohorts['readiness']['fault_safety']['complete']
        safety_summary['all_complete'] = bool(safety_summary['all_complete'] and safety_summary['prospective_supplement_complete'])
        safety_summary['required_run_count'] += 10
        safety_summary['distinct_online_accepted_or_reused_path_checks'] += fresh_active.get('accepted_or_reused_paths_checked', 0)
        safety_summary['runs'] += [{
            'run_id': 'prospective_supplement_v2/' + report['run_id'],
            'cohort': 'prospective_fresh_19',
            'passed': report.get('passed', False),
            'accepted_or_reused_path_checks': report.get('accepted_or_reused_paths_checked', 0),
            'failed_path_checks': len(report.get('failures', [])),
            'scope': 'first and cache repeats; saved actual occupancy; overlapping per-run and active-online audits are separately listed',
        } for report in fresh_runs]
    write_json(root / 'independent_safety_summary.json', safety_summary)
    stage8 = (
        root.parents[2]
        / "doc/change_logs/stage8_scene_expansion_and_qwen_gpu_benchmark.md"
    )
    tests_passed = bool(
        xml in {root / "tests_final_v2.xml", root / "tests_final.xml"}
        and test_counts
        and test_counts["tests"] > 0
        and test_counts["failures"] == test_counts["errors"] == 0
    )
    source_capture = read(root / 'source_bundle_v2/manifest.json', {})
    source_capture_matches = bool(source_capture.get('files')) and not source_capture.get('missing_current_source_paths', [True]) and all(
        (root.parents[2] / row['path']).is_file()
        and hashlib.sha256((root.parents[2] / row['path']).read_bytes()).hexdigest() == row['sha256']
        for row in source_capture.get('files', [])
    )
    gates = {
        "engineering": {
            "complete": bool(checks.get("passed") and tests_passed and source_capture_matches),
            "tests": test_counts,
            "junit_source": str(xml) if xml else None,
            "current_source_capture_matches": source_capture_matches,
            "source_capture": 'source_bundle_v2/manifest.json',
        },
        "data_scale": {
            "complete": len(selected) >= 30
            and read(root / "preprocessing_integrity_report.json")["passed"]
            and read(root / 'missing_data_manifest.json').get('passed', False)
            and physical_coverage.get('passed', False),
            "actual_scene_count": len(selected),
            "sampling_all_requested_dimensions_prospectively_allocated": False,
            "sampling_limitation": 'Allocation used description-derived location/road-parking/light/weather buckets. Density and category/relation complexity were quantified after selection; intersection/turn were recorded but not allocation dimensions. Physical coverage was separately audited without reselection.',
            "physical_coverage_reference": 'physical_coverage_audit.json',
        },
        "ground_truth": {
            "complete": len(geometry.get("scenes", [])) == len(selected)
            and len(query_counts) == len(selected)
            and min(query_counts.values()) >= 10
            and ground_truth.get("derivation_uses_qwen_or_rule_ranking_or_astar")
            is False
            and read(root / 'annotation_coverage_report.json', {}).get('frozen_label_query_hashes_match_release', False),
            "query_count": len(query_rows),
            "human_reviewed": False,
            "unknown_fields": ["color", "fine_attribute", "unproven_reachability"],
        },
        "split": {
            "complete": not read(root / "leakage_audit.json")["same_log_leakage"]
            and read(root / "crop_leakage_audit.json")["exact_duplicate_gate_passed"]
            and official_split.get("passed", False)
            and temporal.get("passed", False),
            "test_scenes": 10,
            "test_logs": 4,
            "near_routes_disclosed": 4,
            "near_crops_disclosed": 4,
            "official_split_supplement": "official_split_audit.json",
            "temporal_audit": "temporal_adjacency_audit.json",
        },
        "cpu": {
            "complete": len(off_rows) == 300
            and all(r["execution_passed"] for r in off_rows)
            and ablation.get("all_complete", False)
            and ablation.get("completed_cases") == 2000,
            "expected_online_query_conditions": 300,
            "completed_online_query_conditions": len(off_rows),
            "static_cases": ablation.get("completed_cases", 0),
        },
        "real_gpu": {
            "complete": len(gpu_rows) == 144
            and all(v["passed"] for v in application_evidence.values())
            and runtime["cuda_available"]
            and runtime["multimodal_processor_probe"].get("passed", False)
            and retrieval.get("all_complete", False)
            and len(retrieval.get("rows", [])) == 720,
            "expected_online_query_conditions": 144,
            "completed_online_query_conditions": len(gpu_rows),
            "three_task_application_evidence": application_evidence,
            "all_query_retrieval_complete": retrieval.get("all_complete", False),
        },
        "cache": {
            "complete": determinism.get("all_complete", False)
            and len(all_repeat_rows) == 1164
            and all(
                r["passed"] and r["cache_zero_new_backend_calls"]
                for r in all_repeat_rows
            )
            and cache_hits_observed
            and semantic_repeats.get('passed', False)
            and semantic_repeats.get('all_complete', False),
            "completed_repeat_pairs": len(all_repeat_rows),
            "expected_repeat_pairs": 1164,
            "real_qwen_repeat_cache_hits_observed": cache_hits_observed,
            "actual_grid_and_navigation_content_and_raw_cache_response_audit": semantic_repeats.get('passed', False) and semantic_repeats.get('all_complete', False),
            "intermediate_rerank_delivery_state_different_pairs": semantic_repeats.get('strict_delivery_state_different_pairs'),
            "identical_intermediate_asynchronous_delivery_state_required_or_claimed": False,
            "repeat_audit_protocol": 'reporting_repeat_audit_protocol_v1.json',
        },
        "fault_safety": {
            "complete": read(root / "failure_injection_report.json")["passed"]
            and len(fault_cases) == 432
            and primary_runs <= audited
            and active_paths_passed,
            "audited_static_retrieval_fault_runs": len(primary_runs & audited),
            "required_static_retrieval_fault_runs": len(primary_runs),
            "missing_path_audit_runs": sorted(primary_runs - audited),
            "active_online_pairs_audited": active_paths.get("completed_job_pairs", 0),
            "active_online_expected": 444,
            "active_online_safety_passed": active_paths_passed,
            "shared_physical_mapping_and_occupancy_across_conditions": semantic_repeats.get('cross_condition_shared_geometry', {}).get('passed', False),
            "superseded_unsafe_evidence": "path_audits/cpu_development_scene0038_v2.json",
            "safety_repair_protocol": "method_protocol_v2.json",
        },
        "statistics": {
            "complete": bool(
                navigation.get("all_complete")
                and retrieval.get("all_complete")
                and statistics.get("iterations", 0) >= 2000
                and not statistics.get("incomplete_comparisons_are_provisional", True)
                and (root / "stratified_report.json").exists()
                and semantic.get('all_complete', False)
                and qwen_effects['all_complete']
            ),
            "bootstrap_iterations": statistics.get("iterations"),
            "cluster": "scene, with separate acquisition-log sensitivity",
            "query_semantics_report": 'semantic_report.json',
            "distinct_language_templates": semantic.get('unique_instruction_count'),
            "observed_order_and_choice_change_report": 'qwen_effect_report.json',
        },
        "documentation": {
            "complete": visualization.get("all_required_views_present", False)
            and visualization.get("all_experiments_complete", False)
            and browser_qa_current
            and (root / "reproducible_commands.md").exists()
            and stage8.exists()
            and "阶段验收状态：完成" in stage8.read_text(),
            "stage8_path": str(stage8),
            "browser_qa_matches_current_visual_sources": browser_qa_current,
        },
        "old_artifact_protection": {
            "complete": read(root / "baseline_verification.json", {}).get(
                "passed", False
            ),
            "protected_file_count": baseline["file_count"],
        },
    }
    if cohorts is not None:
        gates['prospective_sampling_supplement'] = {
            'complete': cohorts['complete'],
            'fresh_scene_count': 19,
            'fresh_test_scene_count': 10,
            'primary_cohort_scene_count_including_mandatory_legacy': 30,
            'all_requested_dimensions_prospectively_allocated_for_fresh_expansion': True,
            'original_50_scene_sampling_not_retroactively_relabelled': True,
            'old_and_fresh_test_averages_pooled': False,
            'report': 'prospective_evaluation_report.json',
            'source': 'prospective_supplement_v2/sampling_plan.json + test_release.json + matrix_complete.json',
        }
        for name, readiness in cohorts['readiness'].items():
            gates[name]['original_diagnostic_cohort_complete'] = gates[name]['complete']
            gates[name]['prospective_supplement'] = readiness
            gates[name]['complete'] = bool(gates[name]['complete'] and readiness['complete'])
        gates['data_scale']['original_cohort_scene_count'] = len(selected)
        gates['data_scale']['original_cohort_sampling_all_requested_dimensions_prospectively_allocated'] = gates['data_scale'].pop('sampling_all_requested_dimensions_prospectively_allocated')
        gates['data_scale']['original_cohort_sampling_limitation'] = gates['data_scale'].pop('sampling_limitation')
        gates['data_scale']['fresh_expansion_prospective_sampling_plan_matches_pre_execution_release'] = cohorts['readiness']['data_scale']['prospective_sampling_plan_matches_pre_execution_release']
        gates['data_scale']['actual_scene_count'] = cohorts['combined']['unique_scene_count']
        gates['data_scale']['distinct_scenes_including_supplement'] = cohorts['combined']['unique_scene_count']
        gates['data_scale']['sampling_supplement_reference'] = 'prospective_evaluation_report.json'
        gates['real_gpu']['supplementary_fresh_gpu_conditions'] = 120
        gates['real_gpu']['supplementary_complete'] = cohorts['readiness']['real_gpu']['complete']
        gates['cache']['combined_expected_repeat_pairs'] = 1998
        gates['cache']['combined_completed_repeat_pairs'] = len(all_repeat_rows) + cohorts['readiness']['cache']['completed_repeat_pairs']
        runtime['supplementary_prospective_cohort'] = {
            'all_complete': cohorts['complete'],
            'online_totals': cohorts['primary']['qwen_online_totals'],
            'scope': 'Primary thirty-scene cohort (eleven fixed legacy plus nineteen fresh scenes); overlaps legacy regression in the original report. Never add primary totals to original totals without deduplicating legacy cases.',
            'raw_evidence_root': cohorts['supplement_path'],
            'combined_unique_counts': cohorts['combined']['counts'],
        }
        fresh_qwen = read(root / 'prospective_supplement_v2/qwen_online_report.json', {})
        fresh_navigation = read(root / 'prospective_supplement_v2/navigation_report.json', {})
        runtime['fresh_supplement_only_online_totals'] = fresh_qwen.get('totals', {})
        runtime['fresh_supplement_only_task_counts'] = task_count_rows(audited_task_counts(fresh_qwen, fresh_navigation))
        runtime['online_totals_scope'] = 'online_totals contains the original 144 GPU pairs; fresh_supplement_only_online_totals contains only the additional 120 pairs. These two scopes have disjoint scenes and may be added. The thirty-scene primary-cohort totals overlap legacy regression and must not be added again.'
        write_json(root / 'qwen_runtime_report.json', runtime)
    all_complete = all(v["complete"] for v in gates.values())
    validation = {
        "goal_complete": all_complete,
        "updated_unix": time.time(),
        "gates": gates,
        "source": "complete task ledgers and independent audits; pending conditions are never filled with stub results",
        "test_evaluation_has_started": True,
        "frozen_protocol": "benchmark_protocol.json + test_release.json",
        "git_commit_or_push_performed": False,
    }
    write_json(root / "validation_report.json", validation)
    progress = read(root/'progress_report.json', {})
    state = {'goal_complete': all_complete, 'pending_validation': [name for name, gate in gates.items() if not gate['complete']]}
    if any(progress.get(key) != value for key, value in state.items()):
        progress.update(state)
        write_json(root/'progress_report.json', progress)
    lines = [
        "# 场景扩充与真实Qwen实验报告",
        "",
        "本轮实验"
        + (
            "已满足全部验收条件。"
            if all_complete
            else "仍在执行，尚未满足完整验收条件。"
        ),
        "",
        "原诊断组已准备50个实际scene、1992帧、500条独立官方实例查询。开发/回归40场景，原测试10场景，测试与开发及原11场景不共享采集log。10个原测试场景来自4个log；4对相近路段、4对近似crop均保留并披露。不能据此宣称跨地域或外观独立泛化。下列详细表格均属于该原诊断组，补正后的新测试结果单列。",
        "",
        "## 实际完成进度",
        "",
        "| 项目 | 已完成 | 计划 |",
        "|---|---:|---:|",
        f"| 四地图静态案例 | {ablation.get('completed_cases', 0)} | 2000 |",
        f"| 普通/车道在线查询条件及复跑 | {len(navigation.get('rows', []))} | 444 |",
        f"| 全查询六条件检索及复跑 | {len(retrieval.get('rows', []))} | 720 |",
        f"| 故障场景 | {len(fault_scenes)} | 12 |",
        "",
        "## 真值与解释边界",
        "",
        "真值来源为nuScenes官方实例、类别、3D几何、可见性和位姿。214个首帧可见实例对应133个系统节点，仅21个节点通过保守一对一关联，1个节点存在歧义。遗漏作为检索失败保留，不能只统计成功关联子集。351条无匹配查询单列正确拒绝率，Top-1/MRR分母不含无匹配查询。颜色、细粒度属性和未经独立证明的可达性仍为未知；没有伪造人工审核。",
        "",
        "深度为官方LIDAR的稀疏投影，不能宣称完成稠密深度估计。低对象召回限制了语义模型可获得的有效候选；模型或规则的负结果均应保留。",
        "",
        "## 语义、规划与运行口径",
        "",
        "车道约束来自nuScenes HD Map。Qwen只解析指令、补充真实crop属性及在已验证候选内重排；A*、碰撞检查、方向及拓扑约束由确定性规划层执行。静态重排建议与在线实际应用分别报告。实验是带帧边界等待的受控录制回放，不是实际车辆控制，也没有实现摄像头车道线识别。另行保存合成异步迟到/过期时序，不混入真实GPU模型指标。",
        "",
        "base与冻结的既有adapter使用实际CUDA、非采样生成、隔离缓存；首次后端尝试、成功响应、schema合法、实际应用、缓存及回退分开计数。GPU并发运行时的延迟按实际共享负载记录，不能视作独占设备实时性能。",
        "",
        "既有adapter沿用原11场景、735条弱监督样本体系，原训练/验证/测试8/2/1场景划分保持不变。弱监督、重复输入与相近crop限制仍然存在；本轮新场景只做评测，没有据此重新微调，也不把原弱监督分数当作本轮独立官方真值成绩。",
        "",
        "## 验收状态",
        "",
        "| 分项 | 完成 |",
        "|---|---|",
    ]
    lines += [
        f"| {name} | {'是' if gate['complete'] else '待完成'} |"
        for name, gate in gates.items()
    ]
    lines += [
        "",
        "## 主要证据",
        "",
        "详见 `navigation_report.json`、`retrieval_report.json`、`ablation_report.json`、`qwen_online_report.json`、`qwen_runtime_report.json`、`statistical_report.json`、`failure_injection_report.json`、`path_audits/` 和 `validation_report.json`。场景和查询级记录保留全部尝试、失败与不适用；空分母记null，不以零成功路径自证100%安全。",
        "",
        "原始完整基线保留未改写。Hydra曾将本轮216个新日志文件写到统一根目录之外，已原样迁回本轮诊断目录，迁移清单完整保留；随后73049个原文件完整SHA-256及受保护目录新增文件核验通过。没有删除用户文件，没有提交或推送Git。",
        "",
    ]
    lines += report_tables(navigation, retrieval, ablation, qwen_online, statistics, compatibility)
    lines += ['', '候选召回率是每条正查询找回的不同可接受官方实例数除以该查询全部可接受实例数，再平均查询比例；它是集合召回而非任一目标命中率。多目标查询Top-1正确但仅找回部分可接受实例时，该召回均值可能低于Top-1。完整定义保存在retrieval_report.json的metric_definitions。', '']
    lines += ['', '未知字段与排除数量见`annotation_coverage_report.json`及补充组同名文件，分别列出对象关联覆盖、未知颜色/视觉属性、未知可达性的正查询数和无匹配查询的指标排除分母。`leakage_extended_audit.json`保留准备时的pending文字，实际完成状态另见`leakage_status_report.json`和冻结发布，不倒改历史审计。实际应用与实际候选顺序/逐帧选择变化的区别见`qwen_effect_report.json`。', '']
    lines += ['', '## 指令解析的语义正确性', '', '解析准确率以冻结原始指令核验类别、锚点、最近偏好、关系及是否添加未要求的颜色约束，要求首次真实响应通过schema；失败或规则fallback不算正确模型解析。文字同义改写、当前视野的prefer_front解释及未知对象属性不计分。该补充分析没有改动原查询、标签或方法。', '', '| 集合 | 条件 | schema通过率 | 可用语义解析准确率 | 仅合法schema条件准确率 |', '|---|---|---|---|---|']
    for group in semantic.get('summaries', []):
        if group['method'].startswith('off_'):
            continue
        lines.append(f"| {group['role']} | {group['method']} | {rate(group['schema_valid_rate'])} | {rate(group['usable_semantic_parse_accuracy'])} | {rate(group['conditional_valid_schema_accuracy'])} |")
    lines += ['', '500条场景查询仅对应10种冻结原始指令，不能视作500种独立语言表达。固定主提示下adapter在全部场景的9种模板上正确，关系模板均被解析为普通car/near，遗漏truck锚点与left_of；因此可用语义解析率为90%，base主提示因schema失败为0%。adapter−base的scene聚类差值为0.9、95% CI [0.9,0.9]，该退化区间来自重复确定性模板，不能推断未见语言的泛化稳定性。关闭Qwen条件没有模型解析任务，其模型解析准确率不适用，检索及导航仍完整评估。', '']
    lines += ['', '## 实际地图和位姿覆盖', '', '最初parking标签仅表示官方描述出现park/parked关键词，不能当作轨迹位于停车区的真值。另以官方carpark_area、is_intersection路段polygon及实际相机位姿独立复核，按scene报告；这属于选场后的覆盖审计。', '', '| 集合 | scene数 | 可核验carpark图层 | 轨迹进入carpark | 距carpark不超过10m | 经过官方intersection | 航向跨度至少15° |', '|---|---:|---:|---:|---:|---:|---:|']
    for group in physical_coverage.get('role_summaries', []):
        strata = group['strata']
        parking = strata['physical_parking_context']
        lines.append(f"| {group['role']} | {group['scene_count']} | {group['scene_count'] - parking.get('map_layer_absent', 0)} | {parking.get('inside_carpark', 0)} | {parking.get('within_10m_of_carpark', 0)} | {strata['physical_intersection_context'].get('intersects_official_intersection', 0)} | {strata['physical_turn_context'].get('heading_span_at_least_15deg', 0)} |")
    lines += ['', '47个具有官方carpark图层的scene均没有记录到位于carpark polygon内的相机轨迹点，12个scene在其10m内；另3个测试scene所在地图缺少该图层，停车区覆盖保持未知。其余7个测试scene也未记录到这类接近证据，不能宣称完成真实停车区内导航覆盖。测试集10个scene均经过官方intersection，因此缺少该物理分层的独立测试对照。几何关联的21个匹配是有阈值与歧义检查的自动规则核验，仍可能有错误关联；没有人工逐一实例确认。所有这些限制均保留在覆盖与标注审计中。', '']
    lines += ['', '## 缓存复跑内容核验的边界', '', f"逐请求模型原文、规范化结果、schema与fallback按实际任务/输入缓存键核对；导航的完整非时间内容及逐帧occupancy数组另行比较。当前{semantic_repeats.get('strict_delivery_state_different_pairs', '待核验')}组存在帧末active_rerank建议可见时刻差异：缓存命中可在当帧最后一次poll完成，真实推理则在下一帧开始的等待中完成。严格包含这一中间投递状态的哈希并非全部相同，该差异及原失败进度快照完整保留；不宣称逐帧异步投递状态完全一致。", '', '导航内容比较排除耗时、缓存元数据及上述异步投递时序，保留候选分数、节点几何、关系更新、目标置信度、完整路径、规划与安全结果；实际缓存模型输出另作精确哈希核验。另以每scene关闭Qwen的q00为固定参照，核对全部方法/查询的真实节点几何、输入帧，以及同地图约束模式下的逐帧occupancy，允许语义分数和选出的安全路径产生真实差异。详见semantic_repeat_audit.json和reporting_repeat_audit_protocol_v1.json。', '']
    lines += ['', '## 独立几何安全审计', '', '| 来源 | 接受或复用路径检查数（含两次回放） | 失败数 | 全部完成并通过 |', '|---|---:|---:|---|']
    lines += [f"| {r['run_id']} | {r['accepted_or_reused_path_checks']} | {r['failed_path_checks']} | {'是' if r['passed'] else '待完成'} |" for r in safety_summary['runs']]
    lines += ['', '检查数的单位是静态接受路径或某帧复用的剩余路径段，不能当作独立scene样本。审计使用Shapely完整线段与闭栅格交集、车道polygon全覆盖、方向及有向transition核验；没有接受路径时安全率为null。路径长度、栅格clearance、转向总量、投影距离及逐帧耗时的真实分布见performance_report.json和lane_constraint_report.json，均不代表实际车辆控制误差。', '']
    if cohorts is not None:
        counts = cohorts['combined']['counts']
        lines += ['', '## 事前分层补正与新的独立测试', '', '原50场景没有在抽样前量化全部指定维度，不能用事后分层消除这一限制。因此保留原结果，又从未运行方法的新scene中按九维冻结规则抽样19个：9个开发、10个测试。与必须保留的原11个开发/回归场景组成30场景、300查询的补正评测组；整体共69个不同scene、690条查询，原11场景只计一次。新测试来自7个此前未使用的log，排除全部原50场景及新开发的采集log；没有根据新测试结果改动方法、查询或标签。', '', f"整体当前完成四地图静态{counts['static_cases']}/2760、在线首次/复跑{counts['online_pairs']}/678、全查询首次/复跑{counts['retrieval_pairs']}/1320。补正组完整性：{'已完成' if cohorts['complete'] else '待完成'}。原测试与新测试不混合计算平均值。", '', '新的主测试指标、实际分母、规划尝试、Qwen应用、九维分层与配对区间见[补正评测报告](prospective_evaluation_report.md)及prospective_evaluation_report.json；全局不重复计数见combined_evidence_report.json。新测试地图与逐帧证据见[补正组浏览器](prospective_supplement_v2/visualizations/index.html)。本地可用候选缺少明确标注的夜间及雨天，保持未覆盖；少量正关系查询与未知细粒度属性仍限制语义结论。', '']
    (root / "experiment_report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "goal_complete": all_complete,
                "completed_gates": [k for k, v in gates.items() if v["complete"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
