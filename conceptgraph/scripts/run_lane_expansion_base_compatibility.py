"""Separate development-only compatibility evidence for an existing prompt."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

from conceptgraph.nuscenes.benchmark_fingerprints import file_sha256, implementation_manifest, scene_input_manifest
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.lane_experiment_metrics import qwen_audit_metrics
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import DEFAULT_CONFIG, run_end_to_end_replay, write_json
from conceptgraph.scripts.verify_lane_expansion_paths import PathVerifier
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat


def main():
    root = DEFAULT_ROOT
    run = root / 'experiments/base_prompt_compatibility_dev_v1'
    run.mkdir(exist_ok=False)
    query = next(json.loads(line) for line in (root/'benchmark_queries.jsonl').open() if json.loads(line)['query_id'] == 'scene-0003-q00')
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    for key, value in {
        'dataset_root': str(root/'converted'), 'lane_map_root': str(root/'lane_maps'),
        'lane_constrained': False, 'experiment_audit': True, 'export_benchmark_nodes': True,
        'mapping_device': 'cpu', 'lane_projection_max_distance_m': 8.0,
        'lane_centerline_weight': 0.25, 'lane_direction_tolerance_deg': 75.0,
        'query_text': query['query_text'], 'online.qwen_enabled': True,
        'online.pose_forward_axis': 2, 'online.relation_coordinate_frame': 'ego_heading',
        'qwen_online.enabled': True, 'qwen_online.adapter_path': None,
        'qwen_online.adapter_revision': 'base-model-no-adapter',
        'qwen_online.cache_dir': str(run/'isolated_cache'), 'qwen_online.frame_period_s': 0.0,
        'qwen_online.deterministic_frame_barrier': True, 'qwen_online.prompt_profile': 'stage4',
    }.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    manifest = {'scope': 'development capability evidence only, excluded from the frozen six-method test comparisons', 'scene': 'scene-0003', 'historical_qwen_split': 'train', 'query': query, 'selected_before_this_run': True, 'reason': 'base shows strict schema failures under the adapter training prompt; exercise the already-existing stage4 full-schema prompt on development data without changing the frozen primary protocol', 'primary_protocol_changed': False, 'prompt_profile': 'stage4', 'implementation': implementation_manifest(PROJECT_ROOT), 'inputs': scene_input_manifest(root, 'scene-0003'), 'driver_sha256': file_sha256(Path(__file__)), 'configuration': OmegaConf.to_container(cfg, resolve=True)}
    manifest['sha256'] = content_sha256(manifest)
    write_json(run/'input_manifest.json', manifest)
    backend = QwenVLChat(str(cfg.qwen_online.model_name_or_path), adapter_path=None, torch_dtype=getattr(torch, str(cfg.qwen_online.torch_dtype)), max_image_side=int(cfg.qwen_online.max_image_side))
    phases = {}
    verifier = PathVerifier(root)
    for repeat in ('first', 'cache_repeat'):
        cfg.qwen_online.log_dir = str(run/repeat/'qwen_audit')
        report, signature = run_end_to_end_replay('scene-0003', cfg, run/repeat, write_outputs=True, qwen_backend=backend)
        frames = run/repeat/'end_to_end_replay/scene-0003/frames.jsonl'
        verifier.online(frames)
        tasks = Counter()
        for path in (run/repeat/'qwen_audit').glob('*.json'):
            result = json.loads(path.read_text())['result']
            if result.get('used_qwen') and result.get('output_schema_valid') and not result.get('cache', {}).get('hit'):
                tasks[result['task']] += 1
        phases[repeat] = {'summary': report, 'signature': signature, 'qwen_audit': qwen_audit_metrics(run/repeat/'qwen_audit', report['qwen_backend_call_count']), 'fresh_schema_valid_responses_by_task': dict(tasks)}
    applications = phases['first']['summary']['qwen']
    valid_parse = phases['first']['fresh_schema_valid_responses_by_task'].get('query_parse', 0) > 0
    deterministic = phases['first']['signature'] == phases['cache_repeat']['signature']
    zero_backend = phases['cache_repeat']['qwen_audit']['backend_call_count'] == 0
    passed = valid_parse and applications.get('query_parse_applied', 0) > 0 and deterministic and zero_backend and not verifier.failures
    result = {'passed': passed, 'scope': manifest['scope'], 'source_manifest_sha256': manifest['sha256'], 'query_parse_application_evidence': valid_parse and applications.get('query_parse_applied', 0) > 0, 'applications': applications, 'phases': phases, 'determinism_passed': deterministic, 'cache_repeat_zero_new_backend_calls': zero_backend, 'path_audit': {'checks': verifier.checks, 'failures': verifier.failures, 'passed': not verifier.failures}, 'counts_in_primary_comparison': False, 'primary_base_finetune_v2_schema_failures_preserved': True}
    write_json(run/'complete.json', result)
    write_json(root/'base_prompt_compatibility_report.json', result)
    if not passed:
        raise RuntimeError('separate base compatibility application evidence incomplete')


if __name__ == '__main__':
    main()
