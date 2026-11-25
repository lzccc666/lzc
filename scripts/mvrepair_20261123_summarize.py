"""只汇总冻结回放证据，生成完整候选表和逐帧得失；不调用模型/建图。"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json

from conceptgraph.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
BASE = ROOT/'outputs/results/nuscenes/validation/iterations'
DEST = ROOT/'outputs/reports/multiview_identity_repair_20261123/main'
P = 'mvrepair_20261123_'


def rows(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def write_csv(path, records):
    with path.open('w', newline='') as file:
        out = csv.DictWriter(file, list(records[0]))
        out.writeheader()
        out.writerows(records)


def main():
    sources = [
        ('L4_v1', P, ['A_front', 'C_three_dedup', 'L4_front', 'L4_three']),
        ('U1_v1', P, ['U1_front', 'U1_three']),
        ('L2_v2', P, ['L2_front', 'L2_three']),
        ('lifecycle_20261123_candidate_L1_v1', 'mvrepair_lifecycle_20261123_', ['L1_lifecycle']),
        ('lifecycle_20261123_candidate_L3_v1', 'mvrepair_lifecycle_20261123_', ['L3_physical_hypotheses']),
    ]
    table, inputs = [], {}
    for suffix, prefix, methods in sources:
        run = BASE/(('mvrepair_' if suffix.startswith('lifecycle') else P)+suffix)
        path = run/(prefix+'report.json')
        report = json.loads(path.read_text())
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in report['results']['node_main']:
            if row['method'] in methods:
                table.append(dict(**row, duplicate_per_prediction=row['duplicate_nodes']/row['predictions'],
                    duplicate_per_GT=row['duplicate_nodes']/row['truth'], source_report=str(path)))
    write_csv(DEST/(P+'all_candidate_node_metrics.csv'), table)

    run = BASE/(P+'L4_v1')
    changes, frame_deltas, prediction_evidence = [], [], []
    mechanism = Counter()
    for directory in sorted(run.glob('scene-*')):
        predictions = rows(directory/(P+'predictions.jsonl.gz'))
        evaluation = rows(directory/(P+'evaluation.jsonl.gz'))
        assert len(predictions) == len(evaluation) == 20
        for pred, ev in zip(predictions, evaluation):
            assert pred['frame_idx'] == ev['frame_idx'] and pred['scene'] == ev['scene']
            audit = pred['audit']['L4_three']
            mechanism['physical_hypotheses_removed'] += audit['physical_hypotheses']['removed']
            mechanism['retired_states'] += len(audit['lifecycle']['retired_states'])
            mechanism['front_positions_used'] += sum(d['used'] for d in audit['front']['position_decisions'])
            for method, objects in pred['predictions'].items():
                for item in objects:
                    prediction_evidence.append(dict(scene=pred['scene'], frame_idx=pred['frame_idx'], method=method,
                        node_id=item['node_id'], category=item['category'], center=json.dumps(item['center']),
                        covariance=json.dumps(item.get('position_covariance_diag')),
                        actual_state_last_measurement_timestamp_s=item.get('state_last_measurement_timestamp_s'),
                        actual_state_measurement_age_s=item.get('state_measurement_age_s'),
                        dense_object_timestamp_s=item.get('dense_object_timestamp_s'),
                        output_timestamp_s=item.get('timestamp_s'), prediction_age_s=item.get('prediction_age_s'),
                        position_source=item.get('position_source'), front_position_source_key=json.dumps(item.get('front_position_source_key'))))
            for scope in ['front', 'union']:
                truth = ev['evaluation_only_official'][scope]
                now = ev['result']['L4_three'][scope]
                match_now = {m['truth_index']: m for m in now['metrics']['matches']}
                for reference in ['A_front', 'C_three_dedup', 'L4_front']:
                    old = ev['result'][reference][scope]
                    match_old = {m['truth_index']: m for m in old['metrics']['matches']}
                    gained, lost = sorted(match_now.keys()-match_old.keys()), sorted(match_old.keys()-match_now.keys())
                    frame_deltas.append(dict(scene=ev['scene'], frame_idx=ev['frame_idx'], scope=scope,
                        reference=reference, candidate='L4_three', TP_before=len(match_old), TP_after=len(match_now),
                        predictions_before=len(old['predictions']), predictions_after=len(now['predictions']),
                        duplicates_before=old['metrics']['duplicate_nodes'], duplicates_after=now['metrics']['duplicate_nodes'],
                        gained_GT=len(gained), lost_GT=len(lost)))
                    for status, indices, result, matches in [('gained', gained, now, match_now), ('lost', lost, old, match_old)]:
                        for index in indices:
                            match = matches[index]
                            obj = result['predictions'][match['prediction_index']]
                            changes.append(dict(scene=ev['scene'], frame_idx=ev['frame_idx'], scope=scope,
                                reference=reference, status=status, instance_token=truth[index]['instance_token'],
                                category=truth[index]['category'], visible_cameras=json.dumps(truth[index]['visible_cameras']),
                                node_id=obj['node_id'], matched_error_m=match['distance_m'],
                                prediction_center=json.dumps(obj['center']), truth_center=json.dumps(truth[index]['center'])))
    write_csv(DEST/(P+'L4_all_frame_gains_losses.csv'), frame_deltas)
    write_csv(DEST/(P+'L4_changed_GT_cases.csv'), changes)
    write_csv(DEST/(P+'L4_all_prediction_evidence.csv'), prediction_evidence)
    gain_totals = Counter((r['reference'], r['scope'], r['status']) for r in changes)
    summary = dict(source_report_sha256=inputs, candidate='L4_three', complete_groups=40,
        mechanism_counts=dict(mechanism), gains_losses=[dict(reference=k[0], scope=k[1], status=k[2], count=v) for k, v in sorted(gain_totals.items())],
        no_model_or_mapping_calls=True, no_inference_changes=True,
        scope='same pilot 2 scenes40groups, one log; descriptive debugging only')
    (DEST/(P+'summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
