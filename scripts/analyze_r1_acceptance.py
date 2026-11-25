"""已保存B0/B1/R1的完整证据、附指标区间和未筛选得失案例。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from conceptgraph.completion.metric_acceptance import paired_metric_uncertainty
from conceptgraph.completion.stage9 import PROJECT, digest


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def nearest(predictions, truth):
    choices = [(float(np.linalg.norm(np.asarray(p['center'][:2]) - truth['center'][:2])), i, p)
               for i, p in enumerate(predictions) if p['category'] == truth['category']]
    if not choices:
        return None
    distance, index, prediction = min(choices, key=lambda x: (x[0], x[1]))
    return dict(distance_m=distance, prediction_index=index, prediction=prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(PROJECT / 'outputs/reports'):
        raise ValueError('报告路径必须位于outputs/reports')
    if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
        raise RuntimeError('低于磁盘保护线')
    output.mkdir(parents=True, exist_ok=False)
    root = PROJECT / 'outputs/results/nuscenes/validation/iterations'
    paths = {'B0': root / 'config_quality_B0_full_20261120_v1',
             'B1': root / 'config_quality_B1_full_20261120_v1',
             'R1': root / 'recall_refinement_R1_full_20261120_v2'}
    reports = {m: read(p / 'report.json') for m, p in paths.items()}
    releases = {m: read(p / 'release.json') for m, p in paths.items()}
    fingerprints = {m: read(p / 'input_fingerprints.json') for m, p in paths.items()}
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/completion/metric_acceptance.py',
        PROJECT / 'conceptgraph/completion/config_comparison.py',
        PROJECT / 'scripts/summarize_mapping_refinement.py']
    source_hashes = {str(p): digest(p) for p in sources}
    for p in sources:
        target = output / 'source_snapshot' / p.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    for m in paths:
        if (reports[m]['scenes'], reports[m]['logs'], reports[m]['frames'],
            reports[m]['result']['counts']['truth']) != (38, 11, 1508, 5723):
            raise ValueError('原始运行范围或分母不完整')
        if releases[m]['scenes'] != releases['B1']['scenes'] or fingerprints[m] != fingerprints['B1']:
            raise ValueError('场景或输入清单不同')
        if releases[m]['threshold_m'] != 3. or releases[m]['seed'] != 17:
            raise ValueError('阈值或种子变化')
    scene_reports = defaultdict(list)
    groups, changes, loss_reasons, cases = defaultdict(Counter), Counter(), Counter(), []
    duplicate_frames, record_hashes, saved_counts = [], {}, defaultdict(Counter)
    for scene in releases['B1']['scenes']:
        name = scene['scene']
        rows = {}
        for m, path in paths.items():
            sr = read(path / name / 'report.json')
            frame_path = path / name / 'frames.jsonl.gz'
            record_hashes[str(frame_path)] = digest(frame_path)
            if record_hashes[str(frame_path)] != sr['records_sha256']:
                raise ValueError('原逐帧记录指纹变化')
            scene_reports[m].append(sr)
            with gzip.open(frame_path, 'rt') as stream:
                rows[m] = list(map(json.loads, stream))
            if len(rows[m]) != sr['frames']:
                raise ValueError('场景帧数变化')
        original_path = next(Path(p) for p in releases['B1']['inputs_sha256']
                             if Path(p).parent.name == name and Path(p).name == 'frames.jsonl')
        with original_path.open() as stream:
            original = {r['frame_idx']: r for r in map(json.loads, stream)}
        for old, base, new in zip(rows['B0'], rows['B1'], rows['R1'], strict=True):
            idx, truth = new['frame_idx'], new['evaluation_only_official']
            frame_rows = dict(B0=old, B1=base, R1=new)
            matched = {}
            for m, row in frame_rows.items():
                if row['frame_idx'] != idx or row['evaluation_only_official'] != truth:
                    raise ValueError('原逐帧范围或GT不同')
                if row['gt_in_inference'] is not False or row['input'] != base['input']:
                    raise ValueError('推理隔离标记或输入不同')
                metrics = row['metrics'][m]
                if len(truth) != metrics['truth'] or len(row['predictions'][m]) != metrics['predictions']:
                    raise ValueError('逐帧分母与记录不同')
                matched[m] = {x['truth_index']: x for x in metrics['matches']}
                for k in ('true_positive', 'truth', 'predictions', 'duplicate_nodes'):
                    saved_counts[m][k] += metrics[k]
            if truth != original[idx]['evaluation_only_official']:
                raise ValueError('原输入GT变化')
            pose = np.asarray(original[idx]['camera_pose'])
            for i, obj in enumerate(truth):
                distance = float(np.linalg.norm(np.asarray(obj['center'][:2]) - pose[:2, 3]))
                bucket = '0_20m' if distance < 20 else '20_40m' if distance < 40 else '40m_plus'
                for m in frame_rows:
                    for group in ['category:' + obj['category'], 'range:' + bucket]:
                        groups[(m, group)]['truth'] += 1
                        groups[(m, group)]['matched'] += i in matched[m]
                a, b = i in matched['B1'], i in matched['R1']
                kind = 'both' if a and b else 'gain' if b else 'loss' if a else 'neither'
                changes[kind] += 1
                if kind not in {'gain', 'loss'}:
                    continue
                example = dict(kind=kind, scene=name, frame_idx=idx, truth_index=i,
                    category=obj['category'], instance=obj['instance_token'], range_m=distance,
                    official_center=obj['center'], input=base['input'],
                    records={m: str(paths[m] / name / 'frames.jsonl.gz') for m in paths},
                    saved_matches={m: matched[m].get(i) for m in paths},
                    nearest_same_class={m: nearest(r['predictions'][m], obj) for m, r in frame_rows.items()})
                if kind == 'loss':
                    closest = example['nearest_same_class']['R1']
                    loss_reasons['category:' + obj['category']] += 1
                    loss_reasons['range:' + bucket] += 1
                    reason = ('no_same_class_prediction' if closest is None else 'within_gate_but_assignment_competition'
                              if closest['distance_m'] <= 3 else 'nearest_center_outside_gate')
                    loss_reasons[reason] += 1
                    if closest is not None:
                        loss_reasons['nearest_source:' + closest['prediction'].get('position_source', 'original_geometry')] += 1
                    example['diagnostic'] = reason
                cases.append(example)
            if new['metrics']['R1']['duplicate_nodes'] != base['metrics']['B1']['duplicate_nodes']:
                duplicate_frames.append(dict(scene=name, frame_idx=idx, input=base['input'],
                    counts={m: row['metrics'][m]['duplicate_nodes'] for m, row in frame_rows.items()},
                    predictions={m: row['predictions'][m] for m, row in frame_rows.items()},
                    official_truth=truth))
    for m in paths:
        if any(reports[m]['result']['counts'][k] != value for k, value in saved_counts[m].items()):
            raise ValueError('完整逐帧汇总与原报告不符')
    uncertainty = paired_metric_uncertainty(scene_reports['B1'], scene_reports['R1'])
    primary = read(output.parent / 'comparison_v1/comparison.json')
    for key in ('recall', 'F1', 'IDF1'):
        np.testing.assert_allclose(uncertainty['intervals'][key]['ci_bonferroni'],
            primary['statistics']['ci_bonferroni'][key], atol=1e-12)
    with (output / 'group_recall_B0_B1_R1.csv').open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'group', 'matched', 'truth', 'recall'])
        for (m, group), c in sorted(groups.items()):
            writer.writerow([m, group, c['matched'], c['truth'], c['matched'] / c['truth']])
    with (output / 'scene_and_log_counts.csv').open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'scene', 'log', 'frames', 'TP', 'GT', 'predictions', 'IDTP', 'duplicates'])
        for m, rows in scene_reports.items():
            for row in rows:
                c = row['result']['counts']
                writer.writerow([m, row['scene'], row['log'], row['frames'], c['true_positive'], c['truth'],
                                 c['predictions'], c['IDTP'], c['duplicate_nodes']])
    write(output / 'all_gain_loss_cases.json', cases)
    write(output / 'all_duplicate_change_frames.json', duplicate_frames)
    write(output / 'secondary_uncertainty.json', uncertainty)
    write(output / 'analysis_release.json', dict(source_sha256=source_hashes,
        original_artifact_sha256={str(p / n): digest(p / n) for p in paths.values()
            for n in ('report.json', 'release.json', 'input_fingerprints.json')},
        original_record_sha256=record_hashes, original_runs={k: str(v) for k, v in paths.items()},
        original_input_fingerprints_equal=True, new_inference=False, new_configuration=False,
        evaluation_threshold_m=3., ground_truth_observations=5723,
        test_accessed=False, default_changed=False))
    write(output / 'evidence_summary.json', dict(saved_counts={m: dict(v) for m, v in saved_counts.items()},
        paired_observation_changes=dict(changes), loss_diagnostics=dict(loss_reasons),
        all_gain_loss_case_count=len(cases), duplicate_change_frame_count=len(duplicate_frames),
        interpretation='全部得失案例按原场景/帧顺序保存；自动诊断不是人工审核或因果分解；原失败不删除。'))
    if any(digest(Path(p)) != h for p, h in source_hashes.items()):
        raise ValueError('分析运行期间源码变化')
    print(json.dumps(dict(intervals=uncertainty['intervals'], changes=dict(changes),
        loss_diagnostics=dict(loss_reasons), cases=len(cases)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
