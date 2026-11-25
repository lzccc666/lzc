"""Save actually consumed unified graphs and bounded post-run audit receipts."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from conceptgraph.completion import relation_nav_runtime_20261125 as runtime

impl = runtime.impl


def main():
    begun = time.monotonic()
    runtime.register()
    manifest = impl.read(impl.DATA / 'manifest.json')
    impl.verify(manifest)
    dest = impl.experiment_directory('nuscenes', 'validation', impl.RUN_ID)
    node_run = impl.ROOT / 'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2'
    frozen = {(r['cohort'], r['scene'], r['frame_idx']): r
              for r in impl.rows(impl.DATA / 'method_frames.jsonl')}
    queries = list(impl.rows(dest / 'navigation_tasks.jsonl'))
    query_groups = {}
    for query in queries:
        query_groups.setdefault((query['scene'], query['frame_idx']), []).append(query)
    truths = {(r['scene'], r['frame_idx']): r['truth']
              for r in impl.rows(impl.DATA / 'evaluation_only/frames.jsonl') if r['cohort'] == 'S2'}
    selected_locations = {frozen['S2', q['scene'], q['frame_idx']]['location']
                          for q in queries if q['inference']['selected_id'] is not None}
    map_paths = [impl.ROOT / 'dataset/nuscenes_map_expansion/expansion' / (location + '.json')
                 for location in sorted(selected_locations)]
    impl.write(impl.DATA / 'postrun_inputs_receipt_v1.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        receipt_status='postrun_not_preregistered_hash',
        missing_preregistration_field='engineering HD map paths/SHA omitted from original manifest',
        affects='engineering road region reproducibility; target selection uses no HD map',
        map_sha256={str(p): impl.sha(p) for p in map_paths},
        node_inputs={str(node_run / (cohort + '_frames.jsonl.gz')):
                     impl.sha(node_run / (cohort + '_frames.jsonl.gz')) for cohort in ['R1', 'S2']},
        task_or_threshold_changes=False))
    counts = {}; diagnoses = Counter(); node_inputs = []; actual_graph_hashes = []
    for cohort in ['R1', 'S2']:
        stats = Counter(); state = {}
        output = dest / ('unified_' + cohort + '_current_graphs.jsonl.gz')
        with gzip.open(output, 'wt') as stream:
            for incoming in impl.rows(node_run / (cohort + '_frames.jsonl.gz')):
                runtime.guard(begun)
                frame = frozen[cohort, incoming['scene'], incoming['frame_idx']]
                assert incoming['predictions'] == frame['predictions']
                key = incoming['scene']
                if key not in state:
                    state[key] = ({}, impl.CurrentRelationCache())
                registry, cache = state[key]
                nodes, eligible, indices = runtime.adapt_nodes(incoming['nodes'], registry)
                # Exactly the frozen navigation() relation input, including its
                # render-time source fields. Raw capture provenance is separately
                # preserved below and is not represented as fresh measurement.
                relation_input = {i: impl.RelationNode(i, n.bbox_center,
                    dynamic=n.motion_state == 'dynamic',
                    observation_count=n.metadata.get('num_detections', 1),
                    last_seen_frame=frame['frame_idx'],
                    metadata=dict(timestamp_s=frame['timestamp_s'],
                        source_node_id=n.metadata['frozen_object_id'], motion_state=n.motion_state))
                    for i, n in nodes.items()}
                full = impl.full_current(relation_input, frame_index=frame['frame_idx'],
                    heading=frame['heading'], timestamp_s=frame['timestamp_s'])
                inc = cache.update_snapshot(relation_input, frame_index=frame['frame_idx'],
                    heading=frame['heading'], timestamp_s=frame['timestamp_s'])
                same = impl.canonical_payload(full.edges) == impl.canonical_payload(inc.edges)
                if not same:
                    raise AssertionError('unified_full_incremental_mismatch')
                payload = impl.canonical_payload(full.edges)
                digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                    separators=(',', ':')).encode()).hexdigest()
                record = dict(cohort=cohort, scene=key, frame_idx=frame['frame_idx'],
                    timestamp_s=frame['timestamp_s'], all_current_fields_equal=same,
                    canonical_graph_sha256=digest, edges=[e.to_dict() for e in full.edges],
                    incremental_edges=[e.to_dict() for e in inc.edges],
                    coordinate_frame_source=impl.FRAME_SOURCE,
                    source_timestamp_semantics='frozen navigation graph uses render timestamp; raw measurements are separate provenance',
                    node_source_provenance=[dict(node_id=r['node_id'],
                        motion_time=r['motion_time'], measurement=r['measurement']) for r in incoming['nodes']],
                    selected_eligible_ids=[i for i, ok in eligible.items() if ok],
                    GT_in_method=False)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stats.update(frames=1, equal=int(same), edges=len(full.edges))
                if cohort == 'S2':
                    actual_graph_hashes.append(dict(scene=key, frame_idx=frame['frame_idx'], sha256=digest))
                    matches = impl.object_frame_metrics(frame['predictions'], truths[key, frame['frame_idx']], threshold_m=3.)['matches']
                    binding = {m['prediction_index']: truths[key, frame['frame_idx']][m['truth_index']]['instance_token'] for m in matches}
                    for query in query_groups[key, frame['frame_idx']]:
                        label = query['evaluation_only']
                        if not label['positive']:
                            continue
                        acceptable = set(label['acceptable_instances'])
                        present = [i for i in nodes if binding.get(indices[i]) in acceptable]
                        confirmed = [i for i in present if eligible[i]]
                        selected = query['inference']['selected_id']
                        if label['selected_correct']:
                            reason = 'selected_correct'
                        elif selected is not None:
                            reason = 'selected_wrong'
                        elif not present:
                            reason = 'correct_target_absent_from_3m_graph_association'
                        elif not confirmed:
                            reason = 'correct_target_present_but_not_confirmed_static_evidence'
                        else:
                            reason = 'confirmed_target_not_selected_relation_or_reference_gate'
                        diagnoses[reason] += 1
                        node_inputs.append(dict(task_id=query['task_id'], reason=reason,
                            associated_correct_node_ids=present, confirmed_correct_node_ids=confirmed,
                            inference_status=query['inference']['status']))
        counts[cohort] = dict(stats)
    impl.write(dest / 'unified_graph_consistency_summary.json', dict(
        passed=True, cohorts=counts, same_graph_as_navigation_S2=True,
        actual_navigation_graphs=actual_graph_hashes,
        no_new_configuration=True, no_new_selection=True,
        source_timestamp_semantics='render-time edge snapshot; source raw capture/age separately recorded',
        no_computation_speedup_claim=True, elapsed_s=time.monotonic() - begun))
    impl.jsonl(dest / 'positive_task_failure_diagnostics.jsonl', node_inputs)
    impl.write(impl.REPORT / 'positive_task_failure_summary.json', dict(
        positive_tasks=sum(diagnoses.values()), mutually_exclusive_task_counts=dict(diagnoses),
        no_threshold_changes=True, GT_used_only_for_after_inference_diagnosis=True))
    integration = impl.read(impl.ROOT / 'doc/交接/20261125/integration_manifest_v1.json')
    git_rows = {}
    for line in subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD'], cwd=impl.ROOT).decode().split('\0'):
        if not line:
            continue
        header, path = line.split('\t', 1)
        git_rows[path] = header.split()[2]
    mismatches = []
    for group in ['delivery_files', 'development_files']:
        for item in integration[group]:
            if git_rows.get(item['path']) != item['integrated_git_blob']:
                mismatches.append(dict(group=group, path=item['path']))
    protection = []
    for item in manifest['protection']:
        protection.append(dict(path=item['path'], expected=item['sha256'],
            actual=impl.sha(item['path']), unchanged=impl.sha(item['path']) == item['sha256']))
    impl.verify(manifest)
    source_files = [Path(__file__), impl.ROOT / 'scripts/relation_nav_finalize_20261125.py']
    impl.write(impl.REPORT / 'final_protection_v1.json', dict(
        passed=not mismatches and all(p['unchanged'] for p in protection),
        protection_count=len(protection), protected=protection,
        integration_git_index_records_checked=len(integration['delivery_files']) + len(integration['development_files']),
        integration_blob_mismatches=mismatches,
        frozen_source_and_input_hashes_unchanged=True,
        postprocessing_source_sha256={str(p): impl.sha(p) for p in source_files},
        no_model_or_GPU_or_planner_calls=True, elapsed_s=time.monotonic() - begun))
    for source in source_files:
        target = dest / 'source_snapshot' / source.relative_to(impl.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil_copy = __import__('shutil').copy2
        shutil_copy(source, target)
    if mismatches or not all(p['unchanged'] for p in protection):
        raise AssertionError('final_protection_mismatch')
    print(json.dumps(dict(passed=True, unified_graphs=counts, positive_diagnoses=dict(diagnoses))))


if __name__ == '__main__':
    main()
