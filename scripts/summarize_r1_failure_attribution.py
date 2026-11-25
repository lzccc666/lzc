"""合并完整 R1 漏失、内部状态及深度几何代理，原全范围分母保持不变。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.r1_failure_attribution import state_evidence, summarize, xy_error


BASE = PROJECT / 'outputs/results/nuscenes/validation/iterations'
EVIDENCE = BASE / 'r1_failure_attribution_20261123_v1'
STATES = BASE / 'r1_failure_states_20261123_v1'
DEPTH = BASE / 'r1_failure_depth_geometry_20261123_v1'
R1 = BASE / 'recall_refinement_R1_full_20261120_v2'


def load_rows(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not json.loads((STATES / 'report.json').read_text())['every_prediction_and_audit_exact']:
        raise ValueError('内部重放未闭合')
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    release = json.loads((EVIDENCE / 'release.json').read_text())
    motion_path = OUTPUT / 'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
    motion = {(r['scene'], r['frame_idx']): r['objects'] for r in map(json.loads, motion_path.open())}
    alignment_path = OUTPUT / 'data_manifest/development_causal_inputs_full_001/alignment_manifest.json'
    plans = {(p['camera']['scene'], p['camera']['frame_idx']): p for p in json.loads(alignment_path.read_text())['plans']}
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/slam/r1_failure_attribution.py']
    for source in sources:
        target = dest / 'source_snapshot' / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(dest / 'release.json', dict(mode='full_R1_failure_attribution', evidence_run=str(EVIDENCE), states_run=str(STATES),
        depth_geometry_run=str(DEPTH), sources_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(p): digest(p) for p in [motion_path, alignment_path, EVIDENCE/'preliminary_report.json', STATES/'report.json', DEPTH/'report.json']},
        seed=17, threshold_m=3., diagnostic_box_iou=.25, diagnostic_box_margin_m=.5,
        attribution_priority=['matched', 'one_to_one_competition', 'final_fov', 'accurate_dense_state',
            'accurate_sparse_state', '2D_support_missing', 'depth_support_missing', 'current_position_error'],
        priority_is_diagnostic_not_causal=True, no_new_method=True, new_frontend_calls=0, GT_in_inference=False,
        test_accessed=False, instance_masks_GT_available=False))
    all_rows, competitions, scene_checks = [], [], []
    instance_seen, instance_matched, instance_continuous = set(), set(), set()
    inputs = {}
    stage_cross = defaultdict(Counter)
    for scene in release['scenes']:
        name = scene['scene']
        evidence = load_rows(EVIDENCE/name/'gt_evidence.jsonl.gz')
        state = {r['frame_idx']: r for r in load_rows(STATES/name/'states.jsonl.gz')}
        depth = {(r['frame_idx'], r['truth_index']): r for r in load_rows(DEPTH/name/'depth_geometry.jsonl.gz')}
        frozen = {r['frame_idx']: r for r in load_rows(R1/name/'frames.jsonl.gz')}
        raw_path = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
        by_frame = defaultdict(dict)
        for e in evidence:
            by_frame[e['frame_idx']][e['truth_index']] = e
        rows = []
        for e in evidence:
            idx, ti = e['frame_idx'], e['truth_index']
            truth = frozen[idx]['evaluation_only_official'][ti]
            labels = motion[(name, idx)]
            if [o['annotation_token'] for o in labels] != [o['annotation_token'] for o in frozen[idx]['evaluation_only_official']]:
                raise ValueError('官方运动标签 join 不一致')
            enriched = state_evidence(e, state[idx], truth)
            enriched['depth_geometry_proxy'] = depth[(idx, ti)]
            enriched['official_motion'] = labels[ti]['official_motion']
            enriched['official_attributes'] = labels[ti]['official_attributes']
            plan = plans[(name, idx)]
            camera, lidar = plan['camera'], plan['lidar']
            enriched['sensor_time'] = dict(camera_timestamp_s=camera['timestamp']/1e6,
                lidar_age_s=(camera['timestamp']-lidar['timestamp'])/1e6 if lidar else None,
                official_sample_minus_camera_s=(camera['official_sample_timestamp']-camera['timestamp'])/1e6,
                depth_available=lidar is not None)
            if not e['matched'] and e['final_within_3m']:
                frame = frozen[idx]
                pred_matches = {m['prediction_index']: m['truth_index'] for m in frame['metrics']['R1']['matches']}
                choices = []
                for p in e['final_within_3m']:
                    pi = p['prediction_index']
                    if pi not in pred_matches:
                        raise AssertionError('存在未用合法匹配，评价器最大匹配数异常')
                    other_i = pred_matches[pi]
                    other = by_frame[idx][other_i]
                    shared = sorted({c['raw_index'] for c in e['candidates']} & {c['raw_index'] for c in other['candidates']})
                    node_id = p['node_id']
                    aliases = state[idx]['aliases']
                    current_assoc = [a for a in state[idx]['associations'] if aliases.get(a['node_id'], a['node_id']) == node_id]
                    choices.append(dict(p, assigned_truth_index=other_i, assigned_instance=other['instance_token'],
                        GT_center_distance_xy_m=xy_error(e['center'], other['center']), shared_raw_detection_support=shared,
                        current_dense_associations=current_assoc,
                        current_dense_association_supports_both=any(a['raw_index'] in shared for a in current_assoc)))
                enriched['competition'] = sorted(choices, key=lambda p: p['error_xy_m'])
                competitions.append(dict(scene=name, frame_idx=idx, truth_index=ti, category=e['category'], choices=enriched['competition']))
            rows.append(enriched)
            outcome = 'TP' if e['matched'] else 'FN'
            for key, value in [('evidence_bin', enriched['evidence_bin']), ('reason', enriched['reason']),
                ('current_measurement_accurate', enriched['current_measurement_accurate']),
                ('current_sparse_accurate', enriched['current_sparse_accurate']),
                ('official_motion', enriched['official_motion'])]:
                stage_cross[key + ':' + str(value)][outcome] += 1
            instance_seen.add(e['instance_token'])
            if e['matched']:
                instance_matched.add(e['instance_token'])
        prev, streak = {}, {}
        for idx, frame in sorted(frozen.items()):
            current = {}
            for m in frame['metrics']['R1']['matches']:
                inst = frame['evaluation_only_official'][m['truth_index']]['instance_token']
                node = frame['predictions']['R1'][m['prediction_index']]['node_id']
                key = (inst, node)
                current[key] = streak.get(key, 0) + 1 if prev.get(key) == idx - 1 else 1
                if current[key] >= 3:
                    instance_continuous.add(inst)
            prev, streak = {k: idx for k in current}, current
        target = dest / name
        target.mkdir()
        with gzip.open(target / 'gt_attribution.jsonl.gz', 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        all_rows.extend(rows)
        scene_checks.append(dict(scene=name, frames=len(frozen), GT=len(rows), TP=sum(r['matched'] for r in rows),
            original_GT_and_order_exact=True, replay_predictions_and_audit_exact=True))
        for p in [raw_path, R1/name/'frames.jsonl.gz', EVIDENCE/name/'gt_evidence.jsonl.gz', STATES/name/'states.jsonl.gz', DEPTH/name/'depth_geometry.jsonl.gz']:
            inputs[str(p)] = digest(p)
    if (len(scene_checks), sum(r['frames'] for r in scene_checks), len(all_rows), sum(r['matched'] for r in all_rows)) != (38, 1508, 5723, 2769):
        raise AssertionError('完整范围校验失败')
    summary = summarize(all_rows)
    fn_bins, fn_reasons, mechanisms, competition_counts = Counter(), Counter(), Counter(), Counter()
    depth_cross = defaultdict(Counter)
    for row in all_rows:
        if row['matched']:
            continue
        fn_bins[row['evidence_bin']] += 1
        fn_reasons[row['reason']] += 1
        if row.get('competition'):
            competition_counts['total'] += 1
            competition_counts[row['category']] += 1
            nearest = row['competition'][0]
            competition_counts['nearest_competing_GT_lt1m' if nearest['GT_center_distance_xy_m'] < 1 else 'nearest_competing_GT_1to3m' if nearest['GT_center_distance_xy_m'] < 3 else 'nearest_competing_GT_3m_plus'] += 1
            competition_counts['any_shared_detection_support'] += any(c['shared_raw_detection_support'] for c in row['competition'])
            competition_counts['nearest_shared_detection_support'] += bool(nearest['shared_raw_detection_support'])
            competition_counts['current_dense_source_supports_both_GT'] += any(c['current_dense_association_supports_both'] for c in row['competition'])
        for observation in row['current_measurements']:
            if observation['kind'] == 'dense' and observation['error_xy_m'] > 3:
                mechanisms['FN_with_dense_error_gt3_observations'] += 1
        bad_dense = [o for o in row['current_measurements'] if o['kind'] == 'dense' and o['error_xy_m'] > 3]
        if bad_dense:
            mechanisms['FN_with_dense_error_gt3_GT_frames'] += 1
            if any(o['center_inside_official_box_margin05'] for o in bad_dense):
                mechanisms['FN_dense_error_gt3_center_still_inside_official_box_margin05'] += 1
            if any(o['center_inside_official_box_margin05'] and o['points_inside_official_box_margin05'] / max(o['points'], 1) >= .5 for o in bad_dense):
                mechanisms['FN_dense_error_gt3_inside_box_and_majority_measured_points_inside'] += 1
                mechanisms['inside_box_center_error_category:' + row['category']] += 1
                mechanisms['inside_box_center_error_motion:' + row['official_motion']] += 1
            if all(o['points_inside_official_box_margin05'] == 0 for o in bad_dense):
                mechanisms['FN_dense_error_gt3_selected_points_all_outside_gt_margin05'] += 1
        if row['iou025_detection_count'] and row['depth_bin'] in {'0', '1_to_4'}:
            proxy = row['depth_geometry_proxy']
            for key in ['gt_projected_rect_valid_depth_points', 'current_points_inside_official_3d_box_margin05']:
                value = proxy[key]
                depth_cross[row['depth_bin'] + ':' + key]['0' if value == 0 else '1_to_4' if value < 5 else '5_plus'] += 1
            inside = proxy['current_points_inside_official_3d_box_margin05']
            covered = max((c['raw_mask_points_inside_gt_3d_box_margin05'] for c in proxy['candidates']), default=0)
            if inside >= 5 and covered < 5:
                mechanisms['sparse_or_zero_FN_gtbox_margin05_has5_but_rawmask_has_under5'] += 1
                mechanisms['mask_support_gap_motion:' + row['official_motion']] += 1
            if inside >= 1 and covered == 0:
                mechanisms['sparse_or_zero_FN_gtbox_margin05_has_points_rawmask_has0'] += 1
    dump(dest / 'input_fingerprints.json', inputs)
    dump(dest / 'competition_cases.json', competitions)
    report = dict(summary, complete=True, scenes=38, logs=11, frames=1508, GT=5723, TP=2769, FN=2954,
        recall=2769/5723, fn_primary_support_bins=dict(fn_bins), fn_mutually_exclusive_observed_stage=dict(fn_reasons),
        stage_cross={k: dict(v) for k,v in stage_cross.items()}, competition_cross=dict(competition_counts),
        geometry_mechanism_proxies=dict(mechanisms), sparse_depth_proxy_cross={k:dict(v) for k,v in depth_cross.items()},
        instance_diagnostics=dict(unique_instances=len(instance_seen), at_least_once_matched=len(instance_matched),
            same_prediction_id_three_consecutive_frames=len(instance_continuous), replaces_frame_recall=False),
        scene_checks=scene_checks,
        limitations=['观察到的漏失阶段不能直接等同纯因果责任', '投影框/3D框内点是支持代理，无实例mask真值，无法独立定量归因分割质量',
            '官方GT略晚于相机、深度为过去扫描，仅做自车刚体变换，无逐对象运动补偿',
            '435竞争是交叉现象，不能与一级FN四格重复相加', '38验证/调试场景已参与方法选择，不是独立测试'],
        output_sha256={str(p.relative_to(dest)): digest(p) for p in dest.glob('scene-*/*.gz')})
    dump(dest / 'report.json', report)
    with (dest/'groups.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'GT', 'TP', 'FN', 'recall'])
        for k,v in sorted(summary['groups'].items()):
            writer.writerow([k,v['truth'],v.get('TP',0),v.get('FN',0),v.get('TP',0)/v['truth']])
    print(json.dumps({k: report[k] for k in ['fn_primary_support_bins','fn_mutually_exclusive_observed_stage','stage_cross','competition_cross','geometry_mechanism_proxies','sparse_depth_proxy_cross','instance_diagnostics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
