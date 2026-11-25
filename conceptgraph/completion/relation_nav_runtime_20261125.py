"""Recorded deployment/schema corrections, without changing frozen task policy."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

from conceptgraph.completion import relation_nav_unified_20261125 as impl
from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.selection_map_transfer import EVIDENCE_POLICY
from conceptgraph.scenegraph.relations import RelationNode

ORIGINAL_ROWS = impl.rows
ORIGINAL_ADAPT = impl.adapt_nodes
CLOCKS = {}


def guard(begun):
    if time.monotonic() - begun > 3600:
        raise TimeoutError('bounded_CPU_stage')
    folders = [impl.DATA, impl.REPORT,
               impl.experiment_directory('nuscenes', 'validation', impl.RUN_ID)]
    for folder in folders:
        existing = folder if folder.exists() else folder.parent
        if shutil.disk_usage(existing.resolve()).free < 20 * 1024**3:
            raise OSError('actual_output_volume_below_20GiB')
    size = sum(p.stat().st_size for folder in folders if folder.exists()
               for p in folder.rglob('*') if p.is_file())
    if size > 128 * 1024**2:
        raise OSError('relation_lane_storage_above_128MiB')


def rows(path):
    for row in ORIGINAL_ROWS(path):
        # Schema stores original records once at frame root, not in each node.
        if Path(path).name == 'S2_frames.jsonl.gz' and 'nodes' in row:
            row = copy.deepcopy(row)
            for node in row['nodes']:
                index = node['original_index']
                legacy = row['predictions'][index]
                if node['node_id'] != legacy['node_id']:
                    raise AssertionError('original_index_identity_mismatch')
                node['legacy_prediction'] = legacy
        yield row


def adapt_nodes(records, registry):
    nodes, eligible, indices = ORIGINAL_ADAPT(records, registry)
    by_index = {r.get('original_index', i): r for i, r in enumerate(records)}
    for local, node in nodes.items():
        record = by_index[indices[local]]
        # New measured three-state view is authoritative; legacy motion is never
        # restored. Dense objects additionally retain the historical depth gate.
        if node.bbox_extent is not None:
            depth_ok, reason = target_evidence_decision(node, EVIDENCE_POLICY)
            eligible[local] = bool(eligible[local] and depth_ok)
            node.metadata['dense_evidence_decision'] = reason
        node.metadata['candidate_policy'] = ('raw_confirmed_static_position' if
            node.bbox_extent is None else 'raw_confirmed_static_and_original_dense_evidence')
        if node.motion_state == 'unknown' and eligible[local]:
            raise AssertionError('unknown_must_not_become_static')
    return nodes, eligible, indices


def relation_nodes(frame, registry, first_seen):
    result = {}
    clock = CLOCKS[frame['cohort'], frame['scene']]
    for obj in frame['predictions']:
        stable = obj['node_id']
        registry.setdefault(stable, len(registry))
        first_seen.setdefault(stable, frame['frame_idx'])
        local = registry[stable]
        source_time = obj.get('timestamp_s')
        # Sparse outputs have no last_seen_frame. Resolve exact source capture
        # to its original render group; never claim a stale node was observed now.
        last = obj.get('last_seen_frame')
        if last is None and source_time is not None:
            last = clock.get(float(source_time))
        result[local] = RelationNode(local, obj['center'],
            dynamic=obj.get('motion_state', 'unknown') == 'dynamic',
            observation_count=obj['num_detections'],
            first_seen_frame=first_seen[stable], last_seen_frame=last,
            metadata=dict(timestamp_s=source_time,
                motion_state=obj.get('motion_state', 'unknown'),
                shape_available=obj.get('shape_available', True),
                source_node_id=stable,
                source_observed_frame=last,
                source_frame_resolution='provided_or_exact_timestamp_else_unknown'))
    return result


def register():
    amendment = impl.DATA / 'runtime_amendment_v1.json'
    paths = [Path(__file__), impl.ROOT / 'scripts/relation_nav_runtime_20261125.py']
    source_hashes = {str(p): impl.sha(p) for p in paths}
    if amendment.exists():
        existing = impl.read(amendment)
        if existing['runtime_source_sha256'] != source_hashes:
            raise AssertionError('runtime_source_changed')
    else:
        log = Path('/tmp/relation_nav_unified_20261125_prepare.log')
        if log.exists():
            shutil.copy2(log, impl.REPORT / 'prepare_v1_failure.log')
        impl.write(amendment, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            frozen_manifest_sha256=impl.sha(impl.DATA / 'manifest.json'),
            runtime_source_sha256=source_hashes,
            changes=[
                'guard checks actual symlink-resolved output volumes instead of source checkout filesystem',
                'bind unified nodes to same-frame original depth via original_index; preserve new motion unknown',
                'dense additionally retains original observation/current-depth/support gate; sparse uses raw position confirmation',
                'sparse relation source frame resolved by exact source timestamp, otherwise unknown rather than current render'],
            configuration_count=1, task_labels_unchanged=True, thresholds_unchanged=True,
            evaluation_not_run_before_amendment=True,
            output_volumes={str(p):dict(resolved=str(p.resolve()),free_bytes=shutil.disk_usage(p).free)
                for p in [impl.DATA, impl.REPORT]},
            dense_policy='new static AND raw semantic position confirmation AND original dense evidence',
            sparse_policy='new static AND raw semantic position confirmation',
            old_motion_used_for_selection=False))
    for frame in ORIGINAL_ROWS(impl.DATA / 'method_frames.jsonl'):
        CLOCKS.setdefault((frame['cohort'], frame['scene']), {})[float(frame['timestamp_s'])] = frame['frame_idx']
    impl.guard = guard
    impl.rows = rows
    impl.adapt_nodes = adapt_nodes
    impl.relation_nodes = relation_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['relations', 'navigation'])
    args, extra = parser.parse_known_args()
    register()
    if args.action == 'navigation':
        sys.argv = [sys.argv[0], *extra]
    {'relations': impl.relations, 'navigation': impl.navigation}[args.action]()
    dest = impl.experiment_directory('nuscenes', 'validation', impl.RUN_ID)
    for relative in ['conceptgraph/completion/relation_nav_runtime_20261125.py',
                     'scripts/relation_nav_runtime_20261125.py']:
        target = dest / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(impl.ROOT / relative, target)


if __name__ == '__main__':
    main()
