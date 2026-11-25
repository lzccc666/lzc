"""Correct the R1-only postprocessing metadata receipt; no selection rerun."""
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

from conceptgraph.completion import relation_nav_runtime_20261125 as runtime

impl = runtime.impl


def main():
    begun = time.monotonic()
    runtime.register()
    dest = impl.experiment_directory('nuscenes', 'validation', impl.RUN_ID)
    old_summary = impl.read(dest / 'unified_graph_consistency_summary.json')
    frame_path = impl.ROOT / 'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2/R1_frames.jsonl.gz'
    frozen = {(r['scene'], r['frame_idx']): r for r in runtime.ORIGINAL_ROWS(impl.DATA / 'method_frames.jsonl') if r['cohort'] == 'R1'}
    state = {}; frames = equal = edges = 0
    new_path = dest / 'unified_R1_current_graphs_v2.jsonl.gz'
    with gzip.open(new_path, 'wt') as stream:
        for incoming in runtime.ORIGINAL_ROWS(frame_path):
            runtime.guard(begun)
            frame = frozen[incoming['scene'], incoming['frame_idx']]
            assert incoming['predictions'] == frame['predictions']
            records = copy.deepcopy(incoming['nodes'])
            for record in records:
                original = incoming['predictions'][record['original_index']]
                if record['node_id'] != original['node_id']:
                    raise AssertionError('original_index_identity_mismatch')
                record['legacy_prediction'] = original
            if incoming['scene'] not in state:
                state[incoming['scene']] = ({}, impl.CurrentRelationCache())
            registry, cache = state[incoming['scene']]
            nodes, eligible, indices = runtime.adapt_nodes(records, registry)
            relnodes = {i: impl.RelationNode(i, n.bbox_center,
                dynamic=n.motion_state == 'dynamic',
                observation_count=n.metadata['num_detections'],
                last_seen_frame=frame['frame_idx'], metadata=dict(timestamp_s=frame['timestamp_s'],
                    source_node_id=n.metadata['frozen_object_id'], motion_state=n.motion_state)) for i, n in nodes.items()}
            full = impl.full_current(relnodes, frame_index=frame['frame_idx'], heading=frame['heading'], timestamp_s=frame['timestamp_s'])
            inc = cache.update_snapshot(relnodes, frame_index=frame['frame_idx'], heading=frame['heading'], timestamp_s=frame['timestamp_s'])
            same = impl.canonical_payload(full.edges) == impl.canonical_payload(inc.edges)
            if not same:
                raise AssertionError('corrected_R1_full_incremental_mismatch')
            payload = impl.canonical_payload(full.edges)
            record = dict(cohort='R1', scene=frame['scene'], frame_idx=frame['frame_idx'],
                timestamp_s=frame['timestamp_s'], all_current_fields_equal=same,
                canonical_graph_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
                edges=[e.to_dict() for e in full.edges], incremental_edges=[e.to_dict() for e in inc.edges],
                coordinate_frame_source=impl.FRAME_SOURCE,
                source_timestamp_semantics='render-time graph snapshot; raw capture is separately recorded',
                node_source_provenance=[dict(node_id=r['node_id'], motion_time=r['motion_time'], measurement=r['measurement']) for r in records],
                observation_count_source='same frame original predictions[original_index].num_detections',
                GT_in_method=False)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            frames += 1; equal += int(same); edges += len(full.edges)
    summary = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        cohorts=dict(R1=dict(frames=frames, equal=equal, edges=edges), S2=old_summary['cohorts']['S2']),
        R1_output=str(new_path), S2_unchanged_output=str(dest / 'unified_S2_current_graphs.jsonl.gz'),
        supersedes='v1 unified_R1_current_graphs only: original observation counts were omitted by postprocessing loader',
        original_v1_outputs_preserved=True, actual_navigation_S2_unchanged=True,
        source_sha256={str(Path(__file__)): impl.sha(__file__),
            str(impl.ROOT / 'scripts/relation_nav_graph_correction_20261125.py'): impl.sha(impl.ROOT / 'scripts/relation_nav_graph_correction_20261125.py')},
        no_new_selection=True, no_threshold_or_node_changes=True, elapsed_s=time.monotonic() - begun)
    impl.write(dest / 'unified_graph_consistency_summary_v2.json', summary)
    impl.write(impl.REPORT / 'postprocessing_correction_v2.json', summary)
    for path in summary['source_sha256']:
        source = Path(path); target = dest / 'source_snapshot' / source.relative_to(impl.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    print(json.dumps(summary['cohorts']))


if __name__ == '__main__':
    main()
