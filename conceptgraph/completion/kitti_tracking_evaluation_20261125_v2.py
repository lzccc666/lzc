"""Complete unchanged Tracking metrics with the explicit nominal-time adapter."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time

from conceptgraph.completion import kitti_local_all_20261125 as impl


def main():
    begun = time.monotonic(); manifest = impl.read(impl.DATA / 'manifest.json'); impl.verify_manifest(manifest)
    dest = impl.experiment_directory('kitti', 'validation', impl.RUN_ID)
    mapping = impl.read(dest / 'mapping_summary.json')
    assert impl.sha(dest / 'mapping_frames.jsonl.gz') == mapping['mapping_frames_sha256']
    frames = {(r['sequence'], r['original_frame_id']): r for r in impl.rows(dest / 'mapping_frames.jsonl.gz')}
    evaluations = list(impl.rows(dest / 'evaluation_frames.jsonl.gz'))
    assert len(evaluations) == len(frames) == 1030
    assert {(r['sequence'], r['original_frame_id']) for r in evaluations} == set(frames)
    source_paths = [Path(__file__), impl.ROOT / 'scripts/kitti_tracking_evaluation_20261125_v2.py']
    impl.write(impl.DATA / 'evaluation_runtime_amendment_v2.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        reason='v1 identity metric adapter omitted required camera_timestamp_s alias; inference and all1030 object evaluations had finished',
        change='identity adapter camera_timestamp_s=already frozen nominal_timestamp_s; not measured sensor time',
        original_failure_log=str(impl.REPORT / 'evaluation_v1.log'),
        source_sha256={str(p): impl.sha(p) for p in source_paths},
        configuration_or_matching_or_denominator_changes=False, inference_rerun=False,
        original_completed_evaluation_file_sha256=impl.sha(dest / 'evaluation_frames.jsonl.gz')))
    totals = {method: Counter() for method in impl.METHODS}
    per_seq = defaultdict(lambda: {m: Counter() for m in impl.METHODS})
    per_class = defaultdict(lambda: {m: Counter() for m in impl.METHODS})
    by_sequence = defaultdict(list); excluded = Counter()
    for evaluated in evaluations:
        frame = frames[evaluated['sequence'], evaluated['original_frame_id']]
        gt = evaluated['official_truth']; excluded.update(o['exclusion_reason'] for o in evaluated['excluded'])
        for method in impl.METHODS:
            metric = evaluated['metrics'][method]
            values = {key: metric[key] for key in ['predictions', 'truth', 'true_positive', 'unmatched_predictions', 'false_negative', 'duplicate_nodes']}
            values['center_error_sum_m'] = sum(m['distance_m'] for m in metric['matches'])
            totals[method].update(values); per_seq[frame['sequence']][method].update(values)
            for category in impl.old.CATEGORIES.values():
                part = impl.object_frame_metrics([o for o in frame['predictions'][method] if o['category'] == category],
                    [o for o in gt if o['category'] == category], threshold_m=3.)
                per_class[category][method].update({key: part[key] for key in ['predictions', 'truth', 'true_positive', 'duplicate_nodes']})
        by_sequence[frame['sequence']].append(dict(frame_idx=frame['frame_idx'],
            camera_timestamp_s=frame['nominal_timestamp_s'],
            sensor_timestamp_measured=False, predictions=frame['predictions'],
            evaluation_only_official=gt, metrics=evaluated['metrics']))
    identity_rows = []
    for sequence, records in by_sequence.items():
        for method in impl.METHODS:
            prepared = impl.old.prepare_sequence(records, method)
            returned = impl.old.return_diagnostics(prepared)
            for support in impl.old.SUPPORTS:
                identity = impl.old.identity_counts(prepared, support)
                identity_rows.append(dict(log_token=sequence, sequence=sequence, method=method,
                    support=support, counts=identity['counts'] | returned['counts']))
    identity, identity_by_sequence = impl.old.aggregate_statistics(identity_rows)
    results = {method: dict(counts=dict(count), metrics=impl.old.object_rates(count)) for method, count in totals.items()}
    impl.write(dest / 'identity_rows_v2.json', identity_rows)
    impl.write(dest / 'identity_by_sequence_v2.json', identity_by_sequence)
    impl.write(dest / 'evaluation_summary_v2.json', dict(
        scope=dict(frames=1030, sequences=11, raw_complete_sequences=0), methods=results,
        per_sequence={k: {m: dict(v) for m, v in group.items()} for k, group in per_seq.items()},
        per_class={k: {m: dict(v) for m, v in group.items()} for k, group in per_class.items()},
        identity=identity, excluded_label_reasons=dict(excluded),
        gate='same category initial IMU world XY<=3m maximum-cardinality, fixed complete local denominator',
        source_evaluation_sha256=impl.sha(dest / 'evaluation_frames.jsonl.gz'),
        official_AP=False, official_HOTA=False, measured_time=False, nominal_dt_s=.1,
        dynamic_GT_accuracy_evaluated=False, GT_in_method=False,
        source_cache='same fixed frontend for all methods,1000 priorGPU+30 priorCPU observations',
        limitations=['all actual local windows, not full official sequences',
            'unmatched predictions include DontCare/unsupported/unlabelled regions; matched fraction not complete false-positive precision',
            'nominal time adapter only; no measured synchronization or dynamic speed accuracy',
            'raw multi-frame motion trace not retained by these old method inputs; no claim of new verified-motion method'],
        CPU_wall_s=time.monotonic() - begun))
    impl.verify_manifest(manifest)
    protected = {p: dict(expected=h, actual=impl.sha(p)) for p, h in manifest['protection_sha256'].items()}
    if any(value['expected'] != value['actual'] for value in protected.values()):
        raise AssertionError('original_protection_changed')
    impl.write(impl.REPORT / 'tracking_final_protection_v2.json', dict(passed=True,
        protected=protected, source_frozen_unchanged=True, mapping_frames_unchanged=True,
        complete_frames=1030, GPU_calls=0))
    for source in source_paths:
        target = dest / 'source_snapshot' / source.relative_to(impl.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        __import__('shutil').copy2(source, target)
    print(__import__('json').dumps(results))


if __name__ == '__main__':
    main()
